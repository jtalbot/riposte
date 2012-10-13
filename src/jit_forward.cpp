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

            case TraceOpCode::box:
            case TraceOpCode::unbox:
            case TraceOpCode::brcast:
            case TraceOpCode::olength:
            case TraceOpCode::decodena:
            case TraceOpCode::decodevl:
            case TraceOpCode::length:
            case TraceOpCode::cenv:
            case TraceOpCode::denv:
            case TraceOpCode::lenv:
            case TraceOpCode::gproto:
            case TraceOpCode::gfalse: 
            case TraceOpCode::gtrue:
            UNARY_FOLD_SCAN_BYTECODES(CASE)
                ir.a = forward[ir.a];
                break;

            case TraceOpCode::load: 
            case TraceOpCode::phi:
            case TraceOpCode::glength:
            case TraceOpCode::reshape:
            case TraceOpCode::gather:
            case TraceOpCode::rep:
            case TraceOpCode::alength:
            case TraceOpCode::encode:
            case TraceOpCode::push:
            BINARY_BYTECODES(CASE)
                ir.a = forward[ir.a];
                ir.b = forward[ir.b];
                break;

            case TraceOpCode::store: 
            case TraceOpCode::seq:
            case TraceOpCode::newenv:
            case TraceOpCode::scatter:
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

