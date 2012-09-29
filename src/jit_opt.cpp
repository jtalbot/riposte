#include "jit.h"
#include "ops.h"

JIT::IR JIT::Normalize(IR ir) {
    switch(ir.op) {
        case TraceOpCode::add:
        case TraceOpCode::mul:
        case TraceOpCode::eq:
        case TraceOpCode::neq:
        case TraceOpCode::lor:
        case TraceOpCode::land:
            if(ir.a > ir.b)
                swap(ir.a, ir.b);
            break;
        case TraceOpCode::lt:
            if(ir.a > ir.b) {
                swap(ir.a, ir.b);
                ir.op = TraceOpCode::gt;
            }
            break;
        case TraceOpCode::le:
            if(ir.a > ir.b) {
                swap(ir.a, ir.b);
                ir.op = TraceOpCode::ge;
            }
            break;
        case TraceOpCode::gt:
            if(ir.a > ir.b) {
                swap(ir.a, ir.b);
                ir.op = TraceOpCode::lt;
            }
            break;
        case TraceOpCode::ge:
            if(ir.a > ir.b) {
                swap(ir.a, ir.b);
                ir.op = TraceOpCode::le;
            }
            break;
        default:
            // nothing
            break;
    }
    return ir;
}

JIT::IR JIT::makeConstant(Value const& v) {
    size_t ci = 0;
    if(constantsMap.find(v) != constantsMap.end()) {
        ci = constantsMap[v];
    }
    else {
        ci = constants.size();
        constants.push_back(v);
        constantsMap[v] = ci;
    }
    return IR(TraceOpCode::constant, ci, v.type, Shape::Empty, Shape::Scalar);
}

JIT::IR JIT::ConstantFold(Thread& thread, IR ir) {
    switch(ir.op) {
            #define CASE1(Name, str, Group, func, Cost) \
                case TraceOpCode::Name: \
                    if(code[ir.a].op == TraceOpCode::constant) { \
                        Value v; \
                        Group##Dispatch<Name##VOp>(thread, constants[code[ir.a].a], v); \
                        ir = makeConstant(v); \
                    } \
                    break;
            UNARY_FOLD_SCAN_BYTECODES(CASE1)
            #undef CASE1

            #define CASE2(Name, str, Group, func, Cost) \
                case TraceOpCode::Name: \
                    if(code[ir.a].op == TraceOpCode::constant && code[ir.b].op == TraceOpCode::constant) { \
                        Value v; \
                        Group##Dispatch<Name##VOp>(thread, constants[code[ir.a].a], constants[code[ir.b].a], v); \
                        ir = makeConstant(v); \
                    } \
                    break;
            BINARY_BYTECODES(CASE2)
            #undef CASE2

/*
            #define CASE3(Name, str, Group, func, Cost) \
                case TraceOpCode::Name: \
                    if(code[ir.a].op == TraceOpCode::constant && code[ir.b].op == TraceOpCode::constant && code[ir.c].op == TraceOpCode::constant) { \
                        Value v; \
                        Group##Dispatch<Name##VOp>(thread, constants[code[ir.a].a], constants[code[ir.b].a], constants[code[ir.c].a], v); \
                        ir = makeConstant(v); \
                    } \
                    break;
            TERNARY_BYTECODES(CASE3)
            #undef CASE3
  */          
            case TraceOpCode::gtrue:
            case TraceOpCode::gfalse:
            case TraceOpCode::gproto:
                if(code[ir.a].op == TraceOpCode::constant)
                    ir = IR(TraceOpCode::nop, Type::Nil, Shape::Empty, Shape::Empty);
                break;                
            default:
                // do nothing
                break;
    }
    return ir;
}

JIT::IR JIT::StrengthReduce(IR ir) {
    switch(ir.op) {
        // numeric+0, numeric*1
        // lower pow -> ^0=>1, ^1=>identity, ^2=>mul, ^0.5=>sqrt, ^-1=>1/x
        // logical & FALSE, logical | TRUE
        // eliminate unnecessary casts
        // simplify expressions !!logical, --numeric, +numeric
        // integer reassociation to recover constants, (a+1)-1 -> a+(1-1) 
   
        // should arrange this to fall through successive attempts at lowering...
 
        case TraceOpCode::mod:
            if(ir.type == Type::Integer) {
                // if mod is a power of 2, can replace with a mask.
                if(code[ir.b].op == TraceOpCode::constant && 
                        abs(((Integer const&)constants[code[ir.b].a])[0]) == 1)
                    return IR(TraceOpCode::constant, 0, Type::Integer, Shape::Empty, Shape::Scalar);
                if(code[ir.b].op == TraceOpCode::brcast
                        && code[code[ir.b].a].op == TraceOpCode::constant
                        && abs(((Integer const&)constants[code[code[ir.b].a].a])[0]) == 1)
                    return IR(TraceOpCode::brcast, 0, Type::Integer, ir.in, ir.out);
            }
        break;

        case TraceOpCode::idiv:
            if(ir.type == Type::Integer) {
                // if power of 2, can replace with a shift.
                if(code[ir.b].op == TraceOpCode::constant && 
                        abs(((Integer const&)constants[code[ir.b].a])[0]) == 1)
                    return IR(TraceOpCode::pos, ir.a, Type::Integer, Shape::Empty, Shape::Scalar);
                if(code[ir.b].op == TraceOpCode::brcast
                        && code[code[ir.b].a].op == TraceOpCode::constant
                        && abs(((Integer const&)constants[code[code[ir.b].a].a])[0]) == 1)
                    return IR(TraceOpCode::pos, ir.a, Type::Integer, ir.in, ir.out);
                
            }
        break;

        case TraceOpCode::gather:
            // lower a gather from a scalar to a brcast
            if(code[ir.a].out.length == 1
                && code[ir.b].op == TraceOpCode::brcast
                && code[ir.b].a == 0)
                return IR(TraceOpCode::brcast, ir.a, ir.type, ir.in, ir.out);
         break;

        case TraceOpCode::brcast:
            if(ir.out.length == 1)
                return IR(TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out);
         break;
        default:
        break;
    }
    return ir;
}

double memcost(size_t size) {
    return (size <= (2<<10) ? 0.075 : (size <= (2<<20) ? 0.3 : 1));
}

/*

    Instruction scheduling
    Careful CSE to forwarding to minimize cost

    Lift filters and gathers above other instructions.

*/

JIT::IRRef JIT::Insert(Thread& thread, std::vector<IR>& code, std::tr1::unordered_map<IR, IRRef>& cse, IR ir) {
    ir = StrengthReduce(ConstantFold(thread, Normalize(ir)));

    if(ir.op == TraceOpCode::pos) {
        return ir.a;
    }

    // CSE cost:
    //  length * load cost
    // No CSE cost:
    //  length * op cost + cost of inputs (if CSEd)
    size_t mysize = ir.out.traceLength * (ir.type == Type::Logical ? 1 : 8);
    double csecost = mysize * memcost(mysize);
    double nocsecost = Opcost(code, ir);

    ir.cost = std::min(csecost, nocsecost);

    if(csecost <= nocsecost && cse.find(ir) != cse.end()) {
        //printf("For %s => %f <= %f\n", TraceOpCode::toString(ir.op), csecost, nocsecost);
        // for mutating operations, have to check if there is a possible intervening mutation...
        /*if(ir.op == TraceOpCode::load) {
            for(IRRef i = code.size()-1; i != cse.find(ir)->second; i--) {
                if(code[i].op == TraceOpCode::store && ir.a == code[i].a && code[code[i].b].op != TraceOpCode::constant) {
                    code.push_back(ir);
                    cse[ir] = code.size()-1;
                    return code.size()-1;
                }
            }
        }
        if(ir.op == TraceOpCode::store) {
            for(IRRef i = code.size()-1; i != cse.find(ir)->second; i--) {
                if(code[i].op == TraceOpCode::store && ir.a == code[i].a && code[code[i].b].op != TraceOpCode::constant) {
                    code.push_back(ir);
                    cse[ir] = code.size()-1;
                    return code.size()-1;
                }
            }
        }*/

        return cse.find(ir)->second;
    }
    else {

        // (1) some type of guard strengthening and guard lifting to permit fusion
        // (2) determine fuseable sequences and uses between sequences.
        // (3) from the back to the front, replace uses in the loop with uses outside of the loop,
        //          IF PROFITABLE
        // (4) DCE

        // Do LICM as DCE rather than CSE.

        // want to do selective, cost driven cse...
        // replacing instruction with instruction outside the loop results in potentially high cost.
        // at each point I can choose to use CSE or recompute the instruction we are on.
        
        // But if the entire thing can be CSE'd that's great e.g. seq of maps followed by sum.
        // So want to do a backwards pass?
        // For a given instruction we know all uses. If dead, no need to compute.
        // If we're going to materialize it in the loop anyway, we should lift it out.
        // If we're not going to materialize it in the loop, we should selectively lift it out.

        // Decision has to be made while/after fusion decisions.
        // What would make us materialize in the loop?
        //      Gather from the vector
        //      
        code.push_back(ir);
        cse[ir] = code.size()-1;
        return code.size()-1;
    }
}

double JIT::Opcost(std::vector<IR>& code, IR ir) {
        switch(ir.op) {
            case TraceOpCode::curenv: 
            case TraceOpCode::newenv:
            case TraceOpCode::sload:
            case TraceOpCode::load:
            case TraceOpCode::store:
            case TraceOpCode::sstore:
            case TraceOpCode::kill:
            case TraceOpCode::push:
            case TraceOpCode::pop:
            case TraceOpCode::gproto:
            case TraceOpCode::gtrue: 
            case TraceOpCode::gfalse:
            case TraceOpCode::scatter:
            case TraceOpCode::jmp:
            case TraceOpCode::exit:
            case TraceOpCode::nest:
            case TraceOpCode::loop:
            case TraceOpCode::brcast:
            case TraceOpCode::phi:
            case TraceOpCode::length:
            case TraceOpCode::rep:
            case TraceOpCode::elength:
            case TraceOpCode::slength:
            case TraceOpCode::olength:
            case TraceOpCode::alength:
            case TraceOpCode::lenv:
            case TraceOpCode::denv:
            case TraceOpCode::cenv:
            case TraceOpCode::reshape:
            case TraceOpCode::constant:
            case TraceOpCode::nop:
                return 10000000000000;
                break;
            
            #define CASE1(Name, str, group, func, Cost) \
                case TraceOpCode::Name: \
                    return (ir.in.traceLength * Cost) + code[ir.a].cost; break;
            UNARY_FOLD_SCAN_BYTECODES(CASE1)

            #define CASE2(Name, str, group, func, Cost) \
                case TraceOpCode::Name: \
                    return (ir.in.traceLength * Cost) + code[ir.a].cost + code[ir.b].cost; break;
            BINARY_BYTECODES(CASE2)

            #define CASE3(Name, str, group, func, Cost) \
                case TraceOpCode::Name: \
                    return (ir.in.traceLength * Cost) + code[ir.a].cost + code[ir.b].cost + code[ir.c].cost; break;
            TERNARY_BYTECODES(CASE3)

            case TraceOpCode::seq:
                return ir.out.traceLength * 1 + code[ir.a].cost + code[ir.b].cost;
                break;

            case TraceOpCode::gather:
                return ir.out.traceLength * memcost(code[ir.a].out.traceLength) + code[ir.b].cost;
                break;

            default:
            {
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in Opcost");
            }
    }
            #undef CASE1
            #undef CASE2
            #undef CASE3
}

// in inital trace should do slot load forwarding
// should probably only have loads and stores for environments?
//

JIT::Aliasing JIT::Alias(std::vector<IR> const& code, IRRef i, IRRef j) {
    if(j < i) std::swap(i, j);

    if(i == j) 
        return MUST_ALIAS;
  
    if(code[i].type != code[j].type)
        return NO_ALIAS;

    if((code[i].out.length != code[j].out.length) &&
            code[code[i].out.length].op == TraceOpCode::constant &&
            code[code[j].out.length].op == TraceOpCode::constant) 
        return NO_ALIAS;
    
    if(code[i].op == TraceOpCode::constant && code[j].op == TraceOpCode::constant)
        return NO_ALIAS;

    // newenv can't alias with something that existed before it was created,
    // TODO: be careful of phis
    if(code[j].op == TraceOpCode::newenv)
        return NO_ALIAS;

    return MAY_ALIAS;
    //   load-load alias if both access the same location in memory
    //   store-load alias if both access the same location in memory
    // AND there are no intervening stores that MAY_ALIAS
/*
    if((code[i].op == TraceOpCode::load || code[i].op == TraceOpCode::store) 
            && code[j].op == TraceOpCode::load) {
        Aliasing a1 = Alias(code, code[i].a, code[j].a);
        Aliasing a2 = Alias(code, code[i].b, code[j].b);
        return (a1 == MUST_ALIAS && a2 == MUST_ALIAS) ? MUST_ALIAS
                : (a1 == NO_ALIAS || a2 == NO_ALIAS) ? NO_ALIAS
                : MAY_ALIAS;
    }

    // load-store alias if stored value aliases with load
    if(code[i].op == TraceOpCode::load && code[j].op == TraceOpCode::store) {
        return Alias(code, i, code[j].c);
    }

    // store-store alias if stored values alias
    if(code[i].op == TraceOpCode::store && code[j].op == TraceOpCode::store) {
        return Alias(code, code[i].c, code[j].c);
    }
*/
    // Alias analysis of stores:
        // ALIAS iff:
            // Objs alias AND keys alias
            
            // Objs alias iff same IR node
            // Keys alias iff same IR node
                // Assumes CSE has been done already and constant propogation
        
        // NO_ALIAS iff:
            // Objs don't alias OR keys don't alias
            
            // Objs don't alias iff:
                // One or both is not an environment
                // Both created by different newenvs 
                // Lexical scope is a forest. Prove that their lexical scope chain is different
                //  on any node.
                // One appears in the dynamic chain of the other
                // Escape analysis demonstrates that an newenv'ed environment is never stored
                //  (where it might be read)

            // Values don't alias iff: 
                // Not the same type
                // If numbers, iff numbers don't alias
                // If strings, iff strings don't alias
                
                // Numbers don't alias iff:
                    // Both constants (and not the same)
                    // Math ops don't alias
                    
                    // Math ops don't alias iff:
                        // Ops are the same
                        // op is 1-to-1, one operand aliases and the other doesn't
                
                        // Operands alias if, cycle back up:

                // Strings don't alias iff:
                    // Both constants (and not the same)
                    // prefix doesn't match or suffix doesn't match
} 

JIT::IRRef JIT::FWD(std::vector<IR> const& code, IRRef i, bool& loopCarried) {
    // search backwards for a store or load we can forward to.
    
    // Each store results in one of:
        // NO_ALIAS: keep searching
        // MAY_ALIAS: stop search, emit load
        // MUST_ALIAS: stop search, forward to this store's value

    IR const& load = code[i];

    // PHIs represent a load forwarded to a store in the previous iteration of the loop.
    // could just not forward across the loop boundary, would require a store and load
    // of loop carried variables.

    loopCarried = false;
    bool crossedLoop = false;

    //printf("\nForwarding %d: ", i);

    for(IRRef j = i-1; j > std::max(load.a, load.b); j--) {
        //printf("%d ", j);
        if(code[j].op == TraceOpCode::nest)
            return i;

        if(code[j].op == TraceOpCode::loop) {
            crossedLoop = true;
        }

        if(code[j].op == load.op) {         // handles both load and elength
            Aliasing a1 = Alias(code, code[j].a, code[i].a);
            Aliasing a2 = Alias(code, code[j].b, code[i].b);
            if(a1 == MUST_ALIAS && a2 == MUST_ALIAS) {
                loopCarried = crossedLoop;    
                return j;
            }
        }
        if(code[j].op == TraceOpCode::store) { 
            Aliasing a1 = Alias(code, code[j].a, code[i].a);
            Aliasing a2 = Alias(code, code[j].b, code[i].b);
            if(a1 == MUST_ALIAS && a2 == MUST_ALIAS) {
                loopCarried = crossedLoop; 
                if(load.op == TraceOpCode::load)
                    return code[j].c;
                else if(load.op == TraceOpCode::elength)
                    return code[j].in.length;
                else
                    _error("Unexpected length");
            }
            else if(!(a1 == NO_ALIAS || a2 == NO_ALIAS)) return i;
        }
    }
    return i;
}

JIT::IRRef JIT::DSE(std::vector<IR> const& code, IRRef i, bool& crossedExit) {
    // search backwards for a store to kill
    crossedExit = false;

    // do DSE
    for(IRRef j = i-1; j < code.size(); j--) {
        // don't cross nests 
        if( code[j].op == TraceOpCode::nest ) 
            break;

        // flag if we cross an exit
        if( code[j].op == TraceOpCode::loop || 
                code[j].op == TraceOpCode::load ||
                code[j].op == TraceOpCode::gtrue ||
                code[j].op == TraceOpCode::gfalse ||
                code[j].op == TraceOpCode::gproto) 
            crossedExit = true;

        if(code[j].op == TraceOpCode::load) {
            Aliasing a1 = Alias(code, code[j].a, code[i].a);
            Aliasing a2 = Alias(code, code[j].b, code[i].b);
            if(a1 != NO_ALIAS && a2 != NO_ALIAS) return i;
        }
        if(code[j].op == TraceOpCode::store) { 
            Aliasing a1 = Alias(code, code[j].a, code[i].a);
            Aliasing a2 = Alias(code, code[j].b, code[i].b);
            if(a1 == MUST_ALIAS && a2 == MUST_ALIAS) return j;
        }
    }
    return i;
}

JIT::IRRef JIT::DPE(std::vector<IR> const& code, IRRef i) {
    for(IRRef j = i-1; j < code.size(); j--) {
        // don't cross guards or loop
        if( code[j].op == TraceOpCode::loop || 
                code[j].op == TraceOpCode::nest || 
                code[j].op == TraceOpCode::load || /* because loads have guards in them */
                code[j].op == TraceOpCode::gtrue ||
                code[j].op == TraceOpCode::gfalse ||
                code[j].op == TraceOpCode::gproto) 
            break;

        if(code[j].op == TraceOpCode::push)
            return j;
    }
    return i;
}

JIT::Exit JIT::BuildExit( Snapshot const& snapshot, Reenter const& reenter, size_t index) {
    Exit e = { snapshot, reenter, index };
    return e;
}


JIT::IRRef JIT::EmitOptIR(
            Thread& thread,
            IR ir,
            std::vector<IR>& code, 
            std::vector<IRRef>& forward, 
            std::tr1::unordered_map<IR, IRRef>& cse,
            Snapshot& snapshot) {

    ir.sunk = false;
    ir.in.length = forward[ir.in.length];
    ir.out.length = forward[ir.out.length];

    switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            
            case TraceOpCode::curenv: 
            {
                if(snapshot.stack.size() == 0) {
                    return Insert(thread, code, cse, ir);
                }
                else {
                    return snapshot.stack.back().environment;
                }
            } break;

            case TraceOpCode::newenv: {
                std::tr1::unordered_map<IR, IRRef> tcse;
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                IRRef f = Insert(thread, code, tcse, ir);
                snapshot.memory.insert(f);
                return f;
            } break;

            case TraceOpCode::load: { 
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                Variable v = { ir.a, (int64_t)ir.b };
                code.push_back(ir);
                bool loopCarried;
                IRRef f = FWD(code, code.size()-1, loopCarried);
                if(f != code.size()-1) {
                    code.pop_back();
                }
                else {
                    exits[code.size()-1] = BuildExit( snapshot, ir.reenter, exits.size()-1 );
                }
                return f;
            } break;
            
            case TraceOpCode::elength:
            {
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                Variable v = { ir.a, (int64_t)ir.b };
                code.push_back(ir);
                bool loopCarried;
                IRRef f = FWD(code, code.size()-1, loopCarried);
                if(f != code.size()-1) {
                    code.pop_back();
                }
                return f;
            } break;
            
            case TraceOpCode::store: {
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                ir.c = forward[ir.c];
                Variable v = { ir.a, (int64_t)ir.b };
                IRRef f = ir.c;
                std::tr1::unordered_map<IR, IRRef> tcse;
                IRRef s = Insert(thread, code, tcse, ir);
                snapshot.memory.insert(s);
                if(ir.b < Loop) {
                    code[s].sunk = true;
                }
                bool crossedExit;
                IRRef j = DSE(code, code.size()-1, crossedExit);    
                if(j != code.size()-1) {
                    snapshot.memory.erase(j);
                    if( !crossedExit )
                        code[j].op = TraceOpCode::nop;
                    else
                        code[j].sunk = true;
                }
                return f;
            } break;
            
            // slot store alias analysis is trivial. Index is identical, or it doesn't alias.
            case TraceOpCode::sload: {
                std::map< int64_t, IRRef >::const_iterator i
                    = snapshot.slotValues.find( (int64_t)ir.b );
                if(i != snapshot.slotValues.end()) {
                    return i->second;
                }
                else {
                    std::tr1::unordered_map<IR, IRRef> tcse;
                    IRRef s = Insert(thread, code, tcse, ir); 
                    snapshot.slotValues[ (int64_t)ir.b ] = s; 
                    exits[code.size()-1] = BuildExit( snapshot, ir.reenter, exits.size()-1 );
                    return s;
                }
            } break;

            case TraceOpCode::slength: {
                std::map< int64_t, IRRef >::const_iterator i
                    = snapshot.slotLengths.find( (int64_t)ir.b );
                if(i != snapshot.slotLengths.end()) {
                    return i->second;
                }
                else {
                    std::tr1::unordered_map<IR, IRRef> tcse;
                    IRRef s = Insert(thread, code, tcse, ir); 
                    snapshot.slotLengths[ (int64_t)ir.b ] = s; 
                    return s;
                }
            } break;
            
            case TraceOpCode::sstore: {
                ir.c = forward[ir.c];
                snapshot.slots[ (int64_t)ir.b ] = ir.c;
                snapshot.slotValues[ (int64_t)ir.b ] = ir.c;
                snapshot.slotLengths[ (int64_t)ir.b ] = ir.out.length;
                return ir.c;
            } break;

            case TraceOpCode::lenv: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::newenv)
                    return code[ir.a].a;
                else
                    return Insert(thread, code, cse, ir); 
            } break;

            case TraceOpCode::denv: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::newenv)
                    return code[ir.a].b;
                else
                    return Insert(thread, code, cse, ir); 
            } break;

            case TraceOpCode::cenv: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::newenv)
                    return code[ir.a].c;
                else
                    return Insert(thread, code, cse, ir); 
            } break;

            case TraceOpCode::kill: {
                std::map<int64_t, IRRef>::iterator i = snapshot.slotValues.begin();
                while(i != snapshot.slotValues.end()) {
                    if(i->first <= ir.a) snapshot.slotValues.erase(i++);
                    else ++i;
                }
                i = snapshot.slotLengths.begin();
                while(i != snapshot.slotLengths.end()) {
                    if(i->first <= ir.a) snapshot.slotLengths.erase(i++);
                    else ++i;
                }
                return 0;
            } break;

            case TraceOpCode::push: {
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                StackFrame frame = frames[ir.c];
                frame.environment = ir.a;
                frame.env = ir.b;
                snapshot.stack.push_back(frame);
                return 0;
            } break;
            case TraceOpCode::pop: {
                snapshot.stack.pop_back();
                return 0;
            } break;

            case TraceOpCode::gproto:
            case TraceOpCode::gfalse: 
            case TraceOpCode::gtrue: {
                ir.a = forward[ir.a];
                IRRef f = Insert(thread, code, cse, ir);
                if(f == code.size()-1)
                    exits[code.size()-1] = BuildExit( snapshot, ir.reenter, exits.size()-1 );
                return 0;
            } break;
            case TraceOpCode::scatter: {
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                return Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::brcast:
            case TraceOpCode::olength:
            UNARY_FOLD_SCAN_BYTECODES(CASE) {
                ir.a = forward[ir.a];
                return Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::reshape:
            case TraceOpCode::gather:
            case TraceOpCode::rep:
            case TraceOpCode::alength:
            BINARY_BYTECODES(CASE)
            {
                ir.a = forward[ir.a]; ir.b = forward[ir.b];
                return Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::seq:
            TERNARY_BYTECODES(CASE)
            {
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                return Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::nest:
            case TraceOpCode::jmp:
            case TraceOpCode::loop:
            {
                std::tr1::unordered_map<IR, IRRef> tcse;
                return Insert(thread, code, tcse, ir);
            } break;
            
            CASE(constant)
            {
                return Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::length:
            {
                return code[forward[ir.a]].out.length;
            } break;

            default:
            {
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in EmitOptIR");
            }

            #undef CASE
        }
}

void JIT::sink(std::vector<bool>& marks, IRRef i) 
{
    #define MARK(k) if(marks[i]) { marks[k] = true; }
    #define ROOT { marks[i] = true; }

    IR const& ir = code[i];

        MARK(ir.out.length);
        MARK(ir.in.length);

        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:

            // Roots
            case TraceOpCode::gproto:
            case TraceOpCode::gfalse: 
            case TraceOpCode::gtrue: {
                ROOT;
                MARK(ir.a);
            }   break; 
            
            case TraceOpCode::load: {
                ROOT;
                MARK(ir.a); MARK(ir.b);
            }   break;

            case TraceOpCode::sload:
            case TraceOpCode::nest:
            case TraceOpCode::jmp:
            case TraceOpCode::exit:
            case TraceOpCode::loop:
            case TraceOpCode::curenv: {
                ROOT;
            }   break;
            
            case TraceOpCode::phi: {
                // these are the most complicated. Basically, if it's a true
                // loop carried dependence, we can't sink it.
                // If it's constant in the loop, we can sink it.
                // If an appropriate loop rotation would eliminate the loop dependence
                //  we could also sink it, but we don't handle that now.
                //  LuaJIT looks for allocations on the right since they, by definition
                //  don't depend on earlier iterations, so a loop rotation would eliminate
                //  their LCD.
                if(ir.a != ir.b) {
                    ROOT;
                    MARK(ir.a); MARK(ir.b);
                }
            }   break;
           
            case TraceOpCode::store: {
                // should this ever be a root?
                MARK(ir.a); MARK(ir.b); MARK(ir.c);
            }   break; 
            
            CASE(constant)
            case TraceOpCode::nop:
            case TraceOpCode::slength:
                break;

            case TraceOpCode::seq:
            case TraceOpCode::scatter: 
            case TraceOpCode::newenv:
            TERNARY_BYTECODES(CASE) {
                MARK(ir.a); MARK(ir.b); MARK(ir.c);
            } break;

            case TraceOpCode::reshape:
            case TraceOpCode::gather:
            case TraceOpCode::rep:
            case TraceOpCode::alength:
            case TraceOpCode::elength:
            BINARY_BYTECODES(CASE) {
                MARK(ir.a); MARK(ir.b);
            } break;
            
            case TraceOpCode::sstore: {
                MARK(ir.c);
            } break;

            case TraceOpCode::brcast:
            case TraceOpCode::olength:
            case TraceOpCode::cenv:
            case TraceOpCode::denv:
            case TraceOpCode::lenv:
            UNARY_FOLD_SCAN_BYTECODES(CASE) {
                MARK(ir.a);
            } break;

            case TraceOpCode::length:
            case TraceOpCode::kill:
            case TraceOpCode::push:
            case TraceOpCode::pop: {
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Should not be reached in SINK");
            } break;

            default:
            {
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in sink");
            }

            #undef CASE
        }
}

void JIT::SINK(void) {
    /*
        Goal is to identify allocations that can be sunk.
        Do this by marking things that can't be sunk first...
    */

    std::vector<bool> marks(code.size(), false);

    bool phiChanged = true;

    // What are the roots?
    // -Guards must be computed.

    // Mark phase
    //  This could be much more efficient with a work queue.
    while(phiChanged) {
    
        // upsweep
        for(IRRef i = code.size()-1; i < code.size(); --i) {
            sink(marks, i);
        }

        // downsweep, mark stores of marked allocations??
        // what about global scope assignments?

        phiChanged = false;
        for(IRRef i = code.size()-2; code[i].op == TraceOpCode::phi; --i) {
            if(marks[code[i].a] || marks[code[i].b]) {
                marks[i] = true;
            }
            if(marks[code[i].a] != marks[code[i].b]) {
                phiChanged = true;
                marks[code[i].a] = true;
                marks[code[i].b] = true;
            }
        }
    }

    // Sweep phase
    for(IRRef i = code.size()-1; i < code.size(); --i) {
        code[i].sunk = !marks[i];
    } 
}
