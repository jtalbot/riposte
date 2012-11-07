#include "jit.h"

/* static */
JIT::IR JIT::Forward(
            JIT::IR ir,
            std::vector<JIT::IRRef> const& forward)
{
    ir.in.length = forward[ir.in.length];
    ir.out.length = forward[ir.out.length];

    switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:

            case TraceOpCode::random:            
            case TraceOpCode::nop:
            case TraceOpCode::nest:
            case TraceOpCode::jmp:
            case TraceOpCode::loop:
            case TraceOpCode::kill:
            case TraceOpCode::curenv: 
            case TraceOpCode::pop:
            case TraceOpCode::sload:
            case TraceOpCode::exit:
            CASE(constant)
                break;

            case TraceOpCode::strip:
            case TraceOpCode::box:
            case TraceOpCode::unbox:
            case TraceOpCode::decodena:
            case TraceOpCode::decodevl:
            case TraceOpCode::cenv:
            case TraceOpCode::denv:
            case TraceOpCode::lenv:
            case TraceOpCode::gproto:
            UNARY_FOLD_SCAN_BYTECODES(CASE)
                ir.a = forward[ir.a];
                break;

            case TraceOpCode::geq: 
            case TraceOpCode::recycle:
            case TraceOpCode::attrget:
            case TraceOpCode::load: 
            case TraceOpCode::phi:
            case TraceOpCode::gather1:
            case TraceOpCode::gather:
            case TraceOpCode::rep:
            case TraceOpCode::encode:
            case TraceOpCode::push:
            BINARY_BYTECODES(CASE)
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                break;

            case TraceOpCode::length:
            case TraceOpCode::rlength:
            case TraceOpCode::resize:
            case TraceOpCode::store: 
            case TraceOpCode::seq:
            case TraceOpCode::newenv:
            case TraceOpCode::scatter:
            case TraceOpCode::scatter1:
            TERNARY_BYTECODES(CASE)
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                ir.c = forward[ir.c];
                break;

            default:
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in Forward");
                break;

            #undef CASE
    }
    return ir;
}

void JIT::ForwardSnapshot(Snapshot& s, 
        std::vector<JIT::IRRef> const& forward) {
    // forward exit
    for(size_t i = 0; i < s.stack.size(); i++) {
        s.stack[i].environment = forward[s.stack[i].environment];
        s.stack[i].env = forward[s.stack[i].env];
    }

    for(std::map<int64_t, IRRef>::iterator i = s.slotValues.begin(); i != s.slotValues.end(); ++i)
    {
        i->second = forward[i->second];
    }

    for(std::map<int64_t, IRRef>::iterator i = s.slots.begin(); i != s.slots.end(); ++i)
    {
        i->second = forward[i->second];
    }

    std::set<IRRef> n;
    for(std::set<IRRef>::iterator i = s.memory.begin(); i != s.memory.end(); ++i)
    {
        n.insert(forward[*i]);
    }
    s.memory.swap(n);
}

