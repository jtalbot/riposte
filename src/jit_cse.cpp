
#include "jit.h"

CSE::CSE(std::vector<JIT::IR>& code)
    : code(code) 
{
}

void CSE::run() {
    std::vector<JIT::IRRef> forward(code.size(), 0);

    forward[0] = 0;
    forward[1] = 1;
    forward[2] = 2;

    for(JIT::IRRef i = 0; i < code.size(); ++i) {
        code[i] = JIT::Propogate(code[i], forward);
        forward[i] = eval(i);
    }
}

/*
    Selective CSE to minimize cost
*/

JIT::IRRef CSE::eval(JIT::IRRef i) {

    JIT::IR& ir = code[i];

    // CSE cost:
    //  length * load cost
    // No CSE cost:
    //  length * op cost + cost of inputs (if CSEd)

    size_t mysize = ir.out.traceLength * (ir.type == Type::Logical ? 1 : 8);
    double csecost = mysize * memcost(mysize);
    double nocsecost = opcost(ir);

    ir.cost = std::min(csecost, nocsecost);

    std::tr1::unordered_map<JIT::IR, JIT::IRRef>::const_iterator k = cse.find(ir);
    if(csecost < nocsecost && k != cse.end()) {
        if(k->second < i) {
            // do cse
            ir.op = TraceOpCode::nop;
            return k->second;
        }
    }
    else {
        cse.insert(std::make_pair( ir, i ));
    }
    return i;
}

double CSE::opcost(JIT::IR ir) {
        switch(ir.op) {
            // stuff that can't be forwarded
            case TraceOpCode::newenv:
            case TraceOpCode::load:
            case TraceOpCode::sload:
            case TraceOpCode::store:
            case TraceOpCode::sstore:
            case TraceOpCode::nest:
            case TraceOpCode::jmp:
            case TraceOpCode::exit:
            case TraceOpCode::loop:
                return 0;


            case TraceOpCode::curenv: 
            case TraceOpCode::kill:
            case TraceOpCode::push:
            case TraceOpCode::pop:
            case TraceOpCode::gproto:
            case TraceOpCode::gtrue: 
            case TraceOpCode::gfalse:
            case TraceOpCode::scatter:
            case TraceOpCode::brcast:
            case TraceOpCode::phi:
            case TraceOpCode::rep:
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


void JIT::CSE(std::vector<IR>& code) {
    ::CSE cse(code);
    cse.run();
}

