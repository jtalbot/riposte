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
    bool retry = true;
    while(retry) {
        retry = false;
        switch(ir.op) {
            // numeric+0, numeric*1
            // lower pow -> ^0=>1, ^1=>identity, ^2=>mul, ^0.5=>sqrt, ^-1=>1/x
            // logical & FALSE, logical | TRUE
            // eliminate unnecessary casts
            // simplify expressions !!logical, --numeric, +numeric
            // integer reassociation to recover constants, (a+1)-1 -> a+(1-1) 

            // should arrange this to fall through successive attempts at lowering...
            case TraceOpCode::pow:
                if(ir.type == Type::Double) {
                    if(code[ir.b].op == TraceOpCode::constant && 
                            ((Double const&)constants[code[ir.b].a])[0] == 1) {
                        ir = IR(TraceOpCode::pos, ir.a, Type::Double, ir.in, ir.out);
                    }
                    else if(code[ir.b].op == TraceOpCode::constant && 
                            ((Double const&)constants[code[ir.b].a])[0] == 2) {
                        ir = IR(TraceOpCode::mul, ir.a, ir.a, Type::Double, ir.in, ir.out);
                        retry = true;
                    }
                    else if(code[ir.b].op == TraceOpCode::brcast
                            && code[code[ir.b].a].op == TraceOpCode::constant
                            && ((Double const&)constants[code[code[ir.b].a].a])[0] == 2) {
                        ir = IR(TraceOpCode::mul, ir.a, ir.a, Type::Double, ir.in, ir.out);
                        retry = true;
                    }
                }
                break;

            case TraceOpCode::lor:
                if(ir.a == ir.b || ir.b == FalseRef)
                    ir = IR(TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out);
                else if(ir.a == TrueRef || ir.b == TrueRef)
                    ir = IR(TraceOpCode::pos, TrueRef, ir.type, ir.in, ir.out);
                else if(ir.a == FalseRef || 
                        (code[ir.a].op == TraceOpCode::brcast && code[ir.a].a == FalseRef))
                    ir = IR(TraceOpCode::pos, ir.b, ir.type, ir.in, ir.out);
                else if(ir.b == FalseRef ||
                        (code[ir.b].op == TraceOpCode::brcast && code[ir.b].a == FalseRef))
                    ir = IR(TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out);
                break;

            case TraceOpCode::land:
                if(ir.a == ir.b)
                    ir = IR(TraceOpCode::pos, ir.a, Type::Integer, ir.in, ir.out); 
                else if(ir.a == FalseRef || ir.b == FalseRef)
                    ir = IR(TraceOpCode::pos, FalseRef, ir.type, ir.in, ir.out);
                else if(ir.a == TrueRef || 
                        (code[ir.a].op == TraceOpCode::brcast && code[ir.a].a == TrueRef))
                    ir = IR(TraceOpCode::pos, ir.b, ir.type, ir.in, ir.out);
                else if(ir.b == TrueRef ||
                        (code[ir.b].op == TraceOpCode::brcast && code[ir.b].a == TrueRef))
                    ir = IR(TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out);
                break;

            case TraceOpCode::ifelse:
                if(ir.b == ir.c)
                    ir = IR(TraceOpCode::pos, ir.b, ir.type, ir.in, ir.out);
                break;

            case TraceOpCode::mod:
                if(ir.type == Type::Integer) {
                    // if mod is a power of 2, can replace with a mask.
                    if(code[ir.b].op == TraceOpCode::constant && 
                            abs(((Integer const&)constants[code[ir.b].a])[0]) == 1)
                        ir = IR(TraceOpCode::constant, 0, Type::Integer, Shape::Empty, Shape::Scalar);
                    if(code[ir.b].op == TraceOpCode::brcast
                            && code[code[ir.b].a].op == TraceOpCode::constant
                            && abs(((Integer const&)constants[code[code[ir.b].a].a])[0]) == 1)
                        ir = IR(TraceOpCode::brcast, 0, Type::Integer, ir.in, ir.out);
                }
                break;

            case TraceOpCode::idiv:
                if(ir.type == Type::Integer) {
                    // if power of 2, can replace with a shift.
                    if(code[ir.b].op == TraceOpCode::constant && 
                            abs(((Integer const&)constants[code[ir.b].a])[0]) == 1)
                        ir = IR(TraceOpCode::pos, ir.a, Type::Integer, Shape::Empty, Shape::Scalar);
                    if(code[ir.b].op == TraceOpCode::brcast
                            && code[code[ir.b].a].op == TraceOpCode::constant
                            && abs(((Integer const&)constants[code[code[ir.b].a].a])[0]) == 1)
                        ir = IR(TraceOpCode::pos, ir.a, Type::Integer, ir.in, ir.out);

                }
                break;

            case TraceOpCode::eq:
                if(ir.a == ir.b) {
                    ir = IR(TraceOpCode::brcast, TrueRef, Type::Integer, ir.in, ir.out);
                    retry = true;
                }
                break;

            case TraceOpCode::neq:
                if(ir.a == ir.b) {
                    ir = IR(TraceOpCode::brcast, FalseRef, Type::Integer, ir.in, ir.out);
                    retry = true;
                }
                break;

            case TraceOpCode::gather:
                // lower a gather from a scalar to a brcast
                if(code[ir.a].out.length == 1
                        && code[ir.b].op == TraceOpCode::brcast
                        && code[ir.b].a == 0) {
                    ir = IR(TraceOpCode::brcast, ir.a, ir.type, ir.in, ir.out);
                    retry = true;
                }
                if(ir.b == 0 && code[ir.a].op == TraceOpCode::seq) {
                    ir = IR(TraceOpCode::pos, code[ir.a].a, ir.type, ir.in, ir.out);
                }
                break;

            case TraceOpCode::brcast:
                if(ir.out.length == code[ir.a].out.length) {
                    ir = IR(TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out);
                }
                break;

            case TraceOpCode::gtrue:
                if(ir.a == TrueRef) {
                    ir = IR(TraceOpCode::nop, Type::Nil, Shape::Empty, Shape::Empty);
                }
                break;

            case TraceOpCode::gfalse:
                if(ir.a == FalseRef) {
                    ir = IR(TraceOpCode::nop, Type::Nil, Shape::Empty, Shape::Empty);
                }
                break;

            case TraceOpCode::encode:
                if(ir.b == FalseRef ||
                    (code[ir.b].op == TraceOpCode::brcast && code[ir.b].a == FalseRef)) {
                    ir = IR(TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out);
                }
                if(code[ir.a].op == TraceOpCode::decodevl
                    &&  code[ir.b].op == TraceOpCode::decodena
                    &&  code[ir.a].a == code[ir.b].a) {
                    ir = IR(TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out);
                }
                else if(code[ir.b].op == TraceOpCode::decodena
                    &&  ir.a == code[ir.b].a) {
                    ir = IR(TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out);
                }
                break;

            case TraceOpCode::decodena:
                if(code[ir.a].op == TraceOpCode::constant) {
                    // TODO: actually check for NA in the constant
                    ir = IR(TraceOpCode::pos, FalseRef, Type::Logical, Shape::Scalar, Shape::Scalar);
                }
                else if(code[ir.a].op == TraceOpCode::encode) {
                    ir = IR(TraceOpCode::pos, code[ir.a].b, Type::Logical, Shape::Scalar, Shape::Scalar);
                }
                // our sequence generators can't produce NAs
                else if(code[ir.a].op == TraceOpCode::seq ||
                        code[ir.a].op == TraceOpCode::random ||
                        code[ir.a].op == TraceOpCode::rep) {
                    ir = IR(TraceOpCode::brcast, FalseRef, Type::Logical, ir.in, ir.out);
                    retry = true;
                }
                break;

            case TraceOpCode::phi:
                if(ir.a == ir.b &&
                    code[ir.a].op == TraceOpCode::constant) {
                    ir = IR(TraceOpCode::nop, Type::Nil, Shape::Empty, Shape::Empty);
                }
                break;

            case TraceOpCode::box:
                if(code[ir.a].op == TraceOpCode::constant) {
                    ir = IR( TraceOpCode::constant, code[ir.a].a, Type::Any, Shape::Empty, Shape::Scalar);
                }
                break;

            case TraceOpCode::unbox:
                if(code[ir.a].op == TraceOpCode::constant) {
                    Value const& v = constants[code[ir.a].a];
                    ir = IR( TraceOpCode::constant, code[ir.a].a, v.type, Shape::Empty, Shape::Scalar);
                }

            case TraceOpCode::aslogical:
                if(code[ir.a].type == Type::Logical) {
                    ir = IR( TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out );
                }
                break;

            case TraceOpCode::asdouble:
                if(code[ir.a].type == Type::Double) {
                    ir = IR( TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out );
                }
                break;
            
            case TraceOpCode::asinteger:
                if(code[ir.a].type == Type::Integer) {
                    ir = IR( TraceOpCode::pos, ir.a, ir.type, ir.in, ir.out );
                }
                break;
            default:
                break;
        }
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

JIT::IRRef JIT::Insert(
    Thread& thread, 
    std::vector<IR>& code, 
    std::tr1::unordered_map<IR, IRRef>& cse, 
    Snapshot& snapshot, 
    IR ir) 
{
    //ir = StrengthReduce(ConstantFold(thread, Normalize(ir)));

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

    std::tr1::unordered_map<IR, IRRef>::const_iterator i = cse.find(ir);
    if(csecost < nocsecost && i != cse.end()) {
        //printf("%d: Found a CSE for %s\n", code.size(), TraceOpCode::toString(ir.op));
        //printf("For %s => %f <= %f\n", TraceOpCode::toString(ir.op), csecost, nocsecost);
        return i->second;
    }
    else {
        ir.exit = BuildExit( ir.exit, snapshot );
        
        code.push_back(ir);
        cse[ir] = code.size()-1;

        return code.size()-1;
    }
}

JIT::IRRef JIT::Optimize(Thread& thread, IRRef i) {
    IR& ir = code[i];

    ir = StrengthReduce(ConstantFold(thread, Normalize(ir)));

    if(ir.op == TraceOpCode::pos) {
        return ir.a;
    }
    return i;
}

void JIT::RunOptimize(Thread& thread) {
    std::tr1::unordered_map<IR, IRRef> cse;

    std::vector<IRRef> forward(code.size(), 0);
    forward[0] = 0;
    forward[1] = 1;
    forward[2] = 2;
    forward[3] = 3;

    std::map<IRRef, IRRef> phis;

    for(IRRef i = 0; i < code.size(); ++i) {
        IR& ir = code[i];
        ir = Forward(ir, forward);
        IRRef fwd = Optimize(thread, i);

        if(fwd == i) {

            size_t mysize = ir.out.traceLength * (ir.type == Type::Logical ? 1 : 8);
            double csecost = mysize * memcost(mysize);
            double nocsecost = Opcost(code, ir);

            ir.cost = std::min(csecost, nocsecost);

            std::tr1::unordered_map<IR, IRRef>::const_iterator j = cse.find(ir);
            if(j != cse.end() &&
                csecost < nocsecost &&
                (ir.op == TraceOpCode::constant ||
                    i < Loop || j->second > Loop)) {
                ir.op = TraceOpCode::nop;
                fwd = j->second;
            }
            else {
                cse[ir] = i;
            }
        }

        forward[i] = fwd;
    }
}

double JIT::Opcost(std::vector<IR>& code, IR ir) {
        switch(ir.op) {
            // Things that we can't CSE, we give a cost of zero to.
            case TraceOpCode::random:
            case TraceOpCode::newenv:
            case TraceOpCode::kill:
            case TraceOpCode::push:
            case TraceOpCode::pop:
            case TraceOpCode::jmp:
            case TraceOpCode::exit:
            case TraceOpCode::nest:
            case TraceOpCode::loop:
            case TraceOpCode::store:
            case TraceOpCode::sstore:
            case TraceOpCode::sload:
            case TraceOpCode::load:
                return 0;
                break;

            // Things that we should absolutely CSE
            case TraceOpCode::gather1:
            case TraceOpCode::strip:
            case TraceOpCode::attrget:
            case TraceOpCode::rep:
            case TraceOpCode::curenv: 
            case TraceOpCode::gproto:
            case TraceOpCode::gtrue: 
            case TraceOpCode::gfalse:
            case TraceOpCode::glength:
            case TraceOpCode::scatter:
            case TraceOpCode::scatter1:
            case TraceOpCode::brcast:
            case TraceOpCode::phi:
            case TraceOpCode::length:
            case TraceOpCode::encode:
            case TraceOpCode::decodena:
            case TraceOpCode::decodevl:
            case TraceOpCode::olength:
            case TraceOpCode::alength:
            case TraceOpCode::lenv:
            case TraceOpCode::denv:
            case TraceOpCode::cenv:
            case TraceOpCode::reshape:
            case TraceOpCode::constant:
            case TraceOpCode::box:
            case TraceOpCode::unbox:
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

        if(code[j].op == load.op) {         // handles load, elength, & ena
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
                code[j].op == TraceOpCode::unbox ||
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
                code[j].op == TraceOpCode::unbox ||
                code[j].op == TraceOpCode::gtrue ||
                code[j].op == TraceOpCode::gfalse ||
                code[j].op == TraceOpCode::gproto) 
            break;

        if(code[j].op == TraceOpCode::push)
            return j;
    }
    return i;
}

int64_t JIT::BuildExit( int64_t stub, Snapshot const& snapshot ) {
    if(stub < 0)
        return stub;

    ExitStub const& es = exitStubs[stub];

    // Check if we already have an exit for this instruction.
    std::map<Instruction const*, size_t>::const_iterator i = uniqueExits.find(es.reenter);
    if(i != uniqueExits.end()) {
        return i->second;
    }

    // If not, make a new exit
    Trace tr;
    tr.traceIndex = Trace::traceCount++;
    tr.Reenter = es.reenter;
    tr.InScope = es.inscope;
    tr.counter = 0;
    tr.ptr = 0;
    tr.function = 0;
    tr.root = dest->root;
    tr.snapshot = snapshot;
    dest->exits.push_back(tr);

    return dest->exits.size()-1;
}

void JIT::Kill(Snapshot& snapshot, int64_t a) {
    std::map<int64_t, IRRef>::iterator i = snapshot.slotValues.begin();
    while(i != snapshot.slotValues.end()) {
        if(i->first <= a) snapshot.slotValues.erase(i++);
        else ++i;
    }
    
    i = snapshot.slots.begin();
    while(i != snapshot.slots.end()) {
        if(i->first <= a) snapshot.slots.erase(i++);
        else ++i;
    }
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
                    return Insert(thread, code, cse, snapshot, ir);
                }
                else {
                    return snapshot.stack.back().environment;
                }
            } break;

            case TraceOpCode::newenv: {
                std::tr1::unordered_map<IR, IRRef> tcse;
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                IRRef f = Insert(thread, code, tcse, snapshot, ir);
                snapshot.memory.insert(f);
                return f;
            } break;

            case TraceOpCode::load: { 
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
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
                IRRef f = ir.c;
                std::tr1::unordered_map<IR, IRRef> tcse;
                IRRef s = Insert(thread, code, tcse, snapshot, ir);
                snapshot.memory.insert(s);
                bool crossedExit;
                IRRef j = DSE(code, code.size()-1, crossedExit);    
                if(j != code.size()-1) {
                    snapshot.memory.erase(j);
                    if( !crossedExit )
                        code[j].op = TraceOpCode::nop;
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
                    IRRef s = Insert(thread, code, tcse, snapshot, ir); 
                    snapshot.slotValues[ (int64_t)ir.b ] = s; 
                    return s;
                }
            } break;

            case TraceOpCode::sstore: {
                ir.c = forward[ir.c];
                
                Kill(snapshot, (int64_t)ir.b);
                snapshot.slotValues[ (int64_t)ir.b ] = ir.c;
                snapshot.slots[ (int64_t)ir.b ] = ir.c;
                    
                // if we're storing to the same slot we just loaded from, this is a no op.
                /*if(code[ir.c].op != TraceOpCode::sload ||
                    code[ir.c].a != ir.b) {
                    std::tr1::unordered_map<IR, IRRef> tcse;
                    IRRef s = Insert(thread, code, tcse, snapshot, ir);
                    snapshot.memory.insert(s);
                }*/
                return ir.c;
            } break;

            case TraceOpCode::lenv: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::newenv)
                    return code[ir.a].a;
                else
                    return Insert(thread, code, cse, snapshot, ir); 
            } break;

            case TraceOpCode::denv: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::newenv)
                    return code[ir.a].b;
                else
                    return Insert(thread, code, cse, snapshot, ir); 
            } break;

            case TraceOpCode::cenv: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::newenv)
                    return code[ir.a].c;
                else
                    return Insert(thread, code, cse, snapshot, ir); 
            } break;

            case TraceOpCode::kill: {
                Kill(snapshot, ir.a);
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
                if(snapshot.stack.size() == 0)
                    _error("Push and pop don't match in trace");
                snapshot.stack.pop_back();
                return 0;
            } break;

            case TraceOpCode::gproto:
            case TraceOpCode::gfalse: 
            case TraceOpCode::gtrue: {
                ir.a = forward[ir.a];
                return Insert(thread, code, cse, snapshot, ir);
            } break;
            case TraceOpCode::scatter1:
            case TraceOpCode::scatter: {
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                return Insert(thread, code, cse, snapshot, ir);
            } break;

            case TraceOpCode::box: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::unbox)
                    return code[ir.a].a;

                return Insert(thread, code, cse, snapshot, ir);
            } break;

            case TraceOpCode::unbox: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::box)
                    return code[ir.a].a;
                else {
                    return Insert(thread, code, cse, snapshot, ir);
                }
            } break;

            case TraceOpCode::length: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::box)
                    return code[code[ir.a].a].out.length;
                else {
                    return Insert(thread, code, cse, snapshot, ir);
                }
            } break;

            case TraceOpCode::glength: {
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                if(code[ir.a].op == TraceOpCode::box)
                    return code[code[ir.a].a].out.length;
                else {
                    return Insert(thread, code, cse, snapshot, ir);
                }
            } break;

            case TraceOpCode::decodevl: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::encode)
                    return code[ir.a].a;
                else {
                    return ir.a;
                    //return Insert(thread, code, cse, snapshot, ir);
                }
            } break;

            case TraceOpCode::decodena: {
                ir.a = forward[ir.a];
                if(code[ir.a].op == TraceOpCode::encode)
                    return code[ir.a].b;
                else {
                    return Insert(thread, code, cse, snapshot, ir);
                }
            } break;

            case TraceOpCode::encode: {
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                /*if(code[ir.a].op == TraceOpCode::decodevl
                    &&  code[ir.b].op == TraceOpCode::decodena
                    &&  code[ir.a].a == code[ir.b].a)
                    return code[ir.a].a;
                else if(code[ir.b].op == TraceOpCode::decodena
                    &&  ir.a == code[ir.b].a)
                    return ir.a;
                else {*/
                    return Insert(thread, code, cse, snapshot, ir);
                //}
            } break;

            case TraceOpCode::strip:
            case TraceOpCode::brcast:
            case TraceOpCode::olength:
            UNARY_FOLD_SCAN_BYTECODES(CASE) {
                ir.a = forward[ir.a];
                return Insert(thread, code, cse, snapshot, ir);
            } break;

            case TraceOpCode::attrget:
            case TraceOpCode::reshape:
            case TraceOpCode::gather1:
            case TraceOpCode::gather:
            case TraceOpCode::rep:
            case TraceOpCode::alength:
            BINARY_BYTECODES(CASE)
            {
                ir.a = forward[ir.a]; ir.b = forward[ir.b];
                return Insert(thread, code, cse, snapshot, ir);
            } break;

            case TraceOpCode::seq:
            TERNARY_BYTECODES(CASE)
            {
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                return Insert(thread, code, cse, snapshot, ir);
            } break;

            case TraceOpCode::random: 
            case TraceOpCode::nest:
            case TraceOpCode::jmp:
            case TraceOpCode::loop:
            {
                std::tr1::unordered_map<IR, IRRef> tcse;
                return Insert(thread, code, tcse, snapshot, ir);
            } break;
           
            CASE(constant)
            {
                return Insert(thread, code, cse, snapshot, ir);
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
            case TraceOpCode::unbox: {
                if(ir.exit >= 0) {
                    ROOT;
                }
                MARK(ir.a);
            }   break;
            case TraceOpCode::gproto:
            case TraceOpCode::gfalse: 
            case TraceOpCode::gtrue: {
                ROOT;
                MARK(ir.a);
            }   break; 
          
            case TraceOpCode::glength:  
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
                if(code[ir.a].op == TraceOpCode::box && code[ir.b].op == TraceOpCode::box) {
                    // nothing
                }
                else if(ir.a != ir.b) {
                    ROOT;
                    MARK(ir.a); MARK(ir.b);
                }
            }   break;
           
            case TraceOpCode::store: {
                // should this ever be a root?
                MARK(ir.a); MARK(ir.b); MARK(ir.c);
            }   break; 
           
            case TraceOpCode::random: 
            CASE(constant)
            case TraceOpCode::nop:
                break;

            case TraceOpCode::seq:
            case TraceOpCode::scatter: 
            case TraceOpCode::scatter1: 
            case TraceOpCode::newenv:
            TERNARY_BYTECODES(CASE) {
                MARK(ir.a); MARK(ir.b); MARK(ir.c);
            } break;

            case TraceOpCode::attrget:
            case TraceOpCode::encode:
            case TraceOpCode::reshape:
            case TraceOpCode::gather1:
            case TraceOpCode::gather:
            case TraceOpCode::rep:
            case TraceOpCode::alength:
            BINARY_BYTECODES(CASE) {
                MARK(ir.a); MARK(ir.b);
            } break;
            
            case TraceOpCode::sstore: {
                MARK(ir.c);
            } break;

            case TraceOpCode::strip:
            case TraceOpCode::box:
            case TraceOpCode::length:
            case TraceOpCode::decodevl:
            case TraceOpCode::decodena:
            case TraceOpCode::brcast:
            case TraceOpCode::olength:
            case TraceOpCode::cenv:
            case TraceOpCode::denv:
            case TraceOpCode::lenv:
            UNARY_FOLD_SCAN_BYTECODES(CASE) {
                MARK(ir.a);
            } break;

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
        for(IRRef i = code.size()-1; i > 0; --i) {
            if(code[i].live)
                sink(marks, i);
        }

        // downsweep, mark stores of marked allocations??
        // what about global scope assignments?

        phiChanged = false;
        for(IRRef i = code.size()-2; i > Loop; --i) {
            if(code[i].op == TraceOpCode::phi && code[i].live) {
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
    }

    bool hasNest = false;
    for(IRRef i = code.size()-1; i > 0; --i) {
        hasNest = hasNest || code[i].op == TraceOpCode::nest;
    }

    // Sweep phase
    for(IRRef i = code.size()-1; i > 0; --i) {
        code[i].sunk = !hasNest && !marks[i];
    } 
}
