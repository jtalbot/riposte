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
            case TraceOpCode::repscalar:
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
    if(j > i) std::swap(i, j);

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

    if(code[i].op == TraceOpCode::newenv && code[j].op == TraceOpCode::newenv)
        return NO_ALIAS;

    if(code[i].op == TraceOpCode::newenv && code[j].op == TraceOpCode::curenv)
        return NO_ALIAS;

    if(code[i].op == TraceOpCode::curenv && code[j].op == TraceOpCode::newenv)
        return NO_ALIAS;

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
    return MAY_ALIAS; 
    
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

        if(code[j].op == TraceOpCode::load) {
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
                return code[j].c;
            }
            else if(!(a1 == NO_ALIAS || a2 == NO_ALIAS)) return i;
        }
    }
    return i;
}

JIT::IRRef JIT::DSE(std::vector<IR> const& code, IRRef i) {
    // search backwards for a store to kill

    // do DSE
    for(IRRef j = i-1; j < code.size(); j--) {
        // don't cross guards or loop
        if( code[j].op == TraceOpCode::loop || 
                code[j].op == TraceOpCode::nest || 
                code[j].op == TraceOpCode::gtrue ||
                code[j].op == TraceOpCode::gfalse ||
                code[j].op == TraceOpCode::load ||
                code[j].op == TraceOpCode::gproto) 
            break;

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


void JIT::EmitOptIR(
            Thread& thread,
            IRRef i,
            IR ir,
            std::vector<IR>& code, 
            std::vector<IRRef>& forward, 
            std::map<Variable, IRRef>& loads,
            std::map<Variable, IRRef>& stores,
            std::tr1::unordered_map<IR, IRRef>& cse,
            std::vector<IRRef>& environments,
            std::vector<StackFrame>& frames,
            std::map<Variable, Phi>& phis) {

    if(i >= 2) {
        ir.in.length = forward[ir.in.length];
        ir.out.length = forward[ir.out.length];
    }

    switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            
            case TraceOpCode::curenv: 
            {
                if(frames.size() == 0) {
                    forward[i] = Insert(thread, code, cse, ir);
                }
                else {
                    forward[i] = frames.back().environment;
                }
            } break;

            case TraceOpCode::newenv: {
                std::tr1::unordered_map<IR, IRRef> tcse;
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                forward[i] = Insert(thread, code, tcse, ir);
                environments.push_back(forward[i]);
            } break;

            case TraceOpCode::load: { 
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                Variable v = { ir.a, (int64_t)ir.b };
                code.push_back(ir);
                bool loopCarried;
                forward[i] = FWD(code, code.size()-1, loopCarried);
                if(forward[i] != code.size()-1) {
                    code.pop_back();
                    if(loopCarried)
                        phis[v] = (Phi) {forward[i], forward[i]};
                }
                else {
                    exits[code.size()-1] = BuildExit( environments, frames, stores, reenters[i], exits.size()-1 );
                }
                if(phis.find(v) != phis.end()) {
                    phis[v].b = forward[i];
                }
            } break;
            
            // slot store alias analysis is trivial. Index is identical, or it doesn't alias.
            case TraceOpCode::sload: {
                // scan back for a previous sstore to forward to.
                forward[i] = code.size();
                for(IRRef j = code.size()-1; j < code.size(); j--) {
                    if(code[j].op == TraceOpCode::sstore && code[j].b == ir.b) {
                        forward[i] = code[j].c;
                        break;
                    }
                }
                if(forward[i] == code.size()) {
                    std::tr1::unordered_map<IR, IRRef> tcse;
                    Insert(thread, code, tcse, ir); 
                    exits[code.size()-1] = BuildExit( environments, frames, stores, reenters[i], exits.size()-1 );
                }
            } break;
            case TraceOpCode::elength:
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
            case TraceOpCode::slength: {
                Variable v = { ir.a, (int64_t)ir.b };
                // forward to previous store, if there is one. 
                if(stores.find(v) != stores.end()) {
                    forward[i] = code[stores[v]].out.length;
                }
                else if(loads.find(v) != loads.end()) {
                    forward[i] = code[loads[v]].out.length;
                }
                else {
                    forward[i] = Insert(thread, code, cse, ir);
                }
            } break;
            case TraceOpCode::store: {
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                ir.c = forward[ir.c];
                Variable v = { ir.a, (int64_t)ir.b };
                stores[v] = forward[i] = ir.c;
                std::tr1::unordered_map<IR, IRRef> tcse;
                Insert(thread, code, tcse, ir);
                if(phis.find(v) != phis.end()) {
                    phis[v].b = forward[i];
                }
                IRRef j = DSE(code, code.size()-1);    
                if(j != code.size()-1)
                    code[j].op = TraceOpCode::nop;
            } break;
            case TraceOpCode::sstore: {
                Variable v = { (IRRef)-1, (int64_t)ir.b };
                forward[i] = ir.c = forward[ir.c];
                std::tr1::unordered_map<IR, IRRef> tcse;
                Insert(thread, code, tcse, ir);
                // do DSE
                for(IRRef j = code.size()-2; j < code.size(); j--) {
                    // don't cross guards or loop
                    if( code[j].op == TraceOpCode::loop || 
                        code[j].op == TraceOpCode::gtrue ||
                        code[j].op == TraceOpCode::gfalse ||
                        code[j].op == TraceOpCode::load ||
                        code[j].op == TraceOpCode::gproto) 
                        break;

                    if( code[j].op == TraceOpCode::sstore && ir.b >= code[j].b ) {
                        code[j].op = TraceOpCode::nop;
                    }
                }
            } break;

            case TraceOpCode::lenv: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::newenv)
                    forward[i] = code[ir.a].a;
                else
                    forward[i] = Insert(thread, code, cse, ir); 
            } break;

            case TraceOpCode::denv: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::newenv)
                    forward[i] = code[ir.a].b;
                else
                    forward[i] = Insert(thread, code, cse, ir); 
            } break;

            case TraceOpCode::cenv: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::newenv)
                    forward[i] = code[ir.a].c;
                else
                    forward[i] = Insert(thread, code, cse, ir); 
            } break;

            case TraceOpCode::kill: {
                // do DSE
                for(IRRef j = code.size()-1; j < code.size(); j--) {
                    // don't cross guards or loop
                    if( code[j].op == TraceOpCode::loop || 
                        code[j].op == TraceOpCode::gtrue ||
                        code[j].op == TraceOpCode::gfalse ||
                        code[j].op == TraceOpCode::load ||
                        code[j].op == TraceOpCode::gproto) 
                        break;

                    if( code[j].op == TraceOpCode::sstore && ir.a >= code[j].b ) {
                        code[j].op = TraceOpCode::nop;
                    }
                }
            } break;

            case TraceOpCode::push: {
                frames.push_back(this->frames[i]);
                frames.back().environment = forward[frames.back().environment];
                frames.back().env = forward[frames.back().env];
            } break;
            case TraceOpCode::pop: {
                frames.pop_back();
            } break;

            case TraceOpCode::gproto:
            case TraceOpCode::gfalse: 
            case TraceOpCode::gtrue: {
                ir.a = forward[ir.a];
                forward[i] = Insert(thread, code, cse, ir);
                if(forward[i] == code.size()-1)
                    exits[code.size()-1] = BuildExit( environments, frames, stores, reenters[i], exits.size()-1 );
            } break;
            case TraceOpCode::scatter: {
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                forward[i] = Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::repscalar:
            case TraceOpCode::olength:
            UNARY_FOLD_SCAN_BYTECODES(CASE) {
                ir.a = forward[ir.a];
                forward[i] = Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::gather:
            case TraceOpCode::rep:
            case TraceOpCode::alength:
            BINARY_BYTECODES(CASE)
            {
                ir.a = forward[ir.a]; ir.b = forward[ir.b];
                forward[i] = Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::seq:
            TERNARY_BYTECODES(CASE)
            {
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                forward[i] = Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::nest:
            case TraceOpCode::jmp:
            case TraceOpCode::loop:
            {
                std::tr1::unordered_map<IR, IRRef> tcse;
                forward[i] = Insert(thread, code, tcse, ir);
            } break;
            
            CASE(constant)
            {
                forward[i] = Insert(thread, code, cse, ir);
            } break;

            case TraceOpCode::length:
            {
                forward[i] = code[forward[ir.a]].out.length;
            } break;

            default:
            {
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in EmitOptIR");
            }

            #undef CASE
        }
}
