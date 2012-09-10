#include "jit.h"

void JIT::AssignRegister(size_t src, size_t index) {
    if(code[src].live) {
        IR& ir = code[index];
        ir.live = true;
        if(ir.reg <= 0) {
            if(ir.op == TraceOpCode::constant) {
                ir.reg = 0;
                return;
            }

            Register r = { ir.type, ir.out }; 

            // if there's a preferred register look for that first.
            if(ir.reg < 0) {
                std::pair<std::multimap<Register, size_t>::iterator,std::multimap<Register, size_t>::iterator> ret;
                ret = freeRegisters.equal_range(r);
                for (std::multimap<Register, size_t>::iterator it = ret.first; it != ret.second; ++it) {
                    if(it->second == -ir.reg) {
                        ir.reg = it->second;
                        freeRegisters.erase(it); 
                        return;
                    }
                }
            }

            // if no preferred or preferred wasn't available fall back to any available or create new.
            std::map<Register, size_t>::iterator i = freeRegisters.find(r);
            if(i != freeRegisters.end()) {
                ir.reg = i->second;
                freeRegisters.erase(i);
                return;
            }
            else {
                ir.reg = registers.size();
                registers.push_back(r);
                return;
            }
        }
    }
}

void JIT::PreferRegister(size_t index, size_t share) {
    IR& ir = code[index];
    IR const& sir = code[share];
    ir.live = true;
    if(ir.reg == 0) {
        ir.reg = sir.reg > 0 ? -sir.reg : sir.reg;
    }
}

void JIT::ReleaseRegister(size_t index) {
    IR const& ir = code[index];
    if(ir.reg > 0) {
        //printf("Releasing register %d at %d\n", assignment[index], index);
        freeRegisters.insert( std::make_pair(registers[ir.reg], ir.reg) );
    }
    else if(ir.reg < 0) {
        printf("Preferred at %d\n", index);
        _error("Preferred register never assigned");
    }
}

void JIT::RegisterAssignment(Exit& e) {
    // backwards pass to do register assignment on a node.
    // its register assignment becomes dead, its operands get assigned to registers if not already.
    // have to maintain same memory space on register assignments.

    // lift this out somewhere
    registers.clear();
    Register invalid = {Type::Nil, Shape::Empty};
    registers.push_back(invalid);
    freeRegisters.clear();
 
    // add all register to the freeRegisters list
    for(size_t i = 0; i < registers.size(); i++) {
        freeRegisters.insert( std::make_pair(registers[i], i) );
    }

    // clear all liveness flags
    for(size_t i = 0; i < code.size(); i++) {
        code[i].live = false;
        code[i].reg = 0;
    } 

    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        ReleaseRegister(i);
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            case TraceOpCode::phi: {
                ir.live = true;
                AssignRegister(i, ir.b);
                PreferRegister(ir.a, ir.b);
            } break;
            case TraceOpCode::gproto: 
            case TraceOpCode::gtrue: 
            case TraceOpCode::gfalse: {
                ir.live = true;
                AssignRegister(i, ir.a);
            } break;
            case TraceOpCode::scatter: {
                AssignRegister(i, ir.c);
                AssignRegister(i, ir.a);
                AssignRegister(i, ir.b);
            } break;
            case TraceOpCode::store:
                ir.live = true;
            case TraceOpCode::newenv:
            TERNARY_BYTECODES(CASE)
            {
                AssignRegister(i, ir.c);
                AssignRegister(i, ir.b);
                AssignRegister(i, ir.a);
            } break;
            case TraceOpCode::sstore:
            {
                ir.live = true;
                AssignRegister(i, ir.c);
            } break;
            case TraceOpCode::load:
            case TraceOpCode::elength:
            case TraceOpCode::alength:
            case TraceOpCode::rep:
            case TraceOpCode::seq:
            case TraceOpCode::gather:
            BINARY_BYTECODES(CASE)
            {
                AssignRegister(i, std::max(ir.a, ir.b));
                AssignRegister(i, std::min(ir.a, ir.b));
            } break;
            case TraceOpCode::repscalar:
            case TraceOpCode::olength:
            case TraceOpCode::lenv:
            case TraceOpCode::denv:
            case TraceOpCode::cenv:
            UNARY_FOLD_SCAN_BYTECODES(CASE)
            {
                AssignRegister(i, ir.a);
            } break;
            case TraceOpCode::loop:
            case TraceOpCode::jmp:
            case TraceOpCode::exit:
            case TraceOpCode::nest:
                ir.live = true;
            case TraceOpCode::curenv:
            case TraceOpCode::nop:
            case TraceOpCode::sload:
            case TraceOpCode::slength:
            case TraceOpCode::constant:
                // do nothing
                break;
            default: {
                printf("Unknown op is %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in RegisterAssignment");
            } break;
            #undef CASE
        }
        AssignRegister(i, ir.in.length);
        AssignRegister(i, ir.out.length);
    }
}

