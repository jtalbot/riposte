#include "jit.h"

void JIT::AssignRegister(size_t index) {
    IR& ir = code[index];
    if(ir.reg <= 0) {
        Register r = { ir.type, ir.out }; 

        if( ir.op == TraceOpCode::constant 
                || ir.op == TraceOpCode::unbox ) {
            ir.reg = registers.size();
            registers.push_back(r);
            return;
        }

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

void JIT::PreferRegister(size_t index, size_t share) {
    IR& ir = code[index];
    IR const& sir = code[share];
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

void JIT::AssignSnapshot(Snapshot const& snapshot) {
    for(size_t i = 0; i < snapshot.stack.size(); i++) {
        AssignRegister(snapshot.stack[i].environment);
        AssignRegister(snapshot.stack[i].env);
    }

    for(std::map< int64_t, IRRef >::const_iterator i = snapshot.slotValues.begin();
            i != snapshot.slotValues.end(); ++i) {
        AssignRegister(i->second);
    }
    
    for(std::set< IRRef >::const_iterator i = snapshot.memory.begin();
            i != snapshot.memory.end(); ++i) {
        if(code[*i].sunk) {
            AssignRegister(*i);
        }
    }
}

void JIT::RegisterAssign(IRRef i, IR ir) {

    if(ir.op == TraceOpCode::scatter) {
        // scatter is special since it can resize its register.
        // so resize the register here and then attempt to reuse...
        registers[ir.reg].shape = code[ir.a].out;
    }

    if( ir.op != TraceOpCode::unbox &&
        ir.op != TraceOpCode::constant)
    {
        ReleaseRegister(i);
    }

    switch(ir.op) 
    {
#define CASE(Name, ...) case TraceOpCode::Name:
        case TraceOpCode::phi: 
            {
                AssignRegister(ir.b);
                PreferRegister(ir.a, ir.b);
            } break;
        case TraceOpCode::gproto: 
        case TraceOpCode::gtrue: 
        case TraceOpCode::gfalse: 
            {
                AssignRegister(ir.a);
            } break;
        case TraceOpCode::scatter: 
            {
                PreferRegister(ir.a, i); 
                AssignRegister(ir.a);
                AssignRegister(ir.c);
                AssignRegister(ir.b);
                // necessary to size input
                AssignRegister(code[ir.a].out.length);
            } break;
        case TraceOpCode::store:
        case TraceOpCode::newenv:
            TERNARY_BYTECODES(CASE)
            {
                AssignRegister(ir.c);
                AssignRegister(ir.b);
                AssignRegister(ir.a);
            } break;
        case TraceOpCode::sstore:
            {
                AssignRegister(ir.c);
            } break;
        case TraceOpCode::reshape:
            {
                PreferRegister(ir.b, i);
                AssignRegister(ir.b);
                AssignRegister(ir.a);
            } break;
        case TraceOpCode::load:
            {
                AssignRegister(std::max(ir.a, ir.b));
                AssignRegister(std::min(ir.a, ir.b));
            } break;
        case TraceOpCode::glength:
        case TraceOpCode::encode:
        case TraceOpCode::alength:
        case TraceOpCode::rep:
        case TraceOpCode::seq:
        case TraceOpCode::gather:
            BINARY_BYTECODES(CASE)
            {
                AssignRegister(std::max(ir.a, ir.b));
                AssignRegister(std::min(ir.a, ir.b));
            } break;
        case TraceOpCode::box:
        case TraceOpCode::unbox:
        case TraceOpCode::decodevl:
        case TraceOpCode::decodena:
        case TraceOpCode::length:
        case TraceOpCode::brcast:
        case TraceOpCode::olength:
        case TraceOpCode::lenv:
        case TraceOpCode::denv:
        case TraceOpCode::cenv:
            UNARY_FOLD_SCAN_BYTECODES(CASE)
            {
                AssignRegister(ir.a);
            } break;
        case TraceOpCode::loop:
        case TraceOpCode::jmp:
        case TraceOpCode::exit:
        case TraceOpCode::nest:
        case TraceOpCode::pop:
        case TraceOpCode::curenv:
        case TraceOpCode::nop:
        case TraceOpCode::sload:
        case TraceOpCode::constant:
            // do nothing
            break;
        default: {
                     printf("Unknown op is %s\n", TraceOpCode::toString(ir.op));
                     _error("Unknown op in RegisterAssignment");
                 } break;
#undef CASE
    }

    AssignRegister(ir.in.length);
    AssignRegister(ir.out.length);
}
void JIT::RegisterAssignment() {
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

    // clear all register assignments 
    for(size_t i = 0; i < code.size(); i++) {
        code[i].reg = 0;
    } 

    // assign sunk registers first, including any thing captured in a snapshot
    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        if(ir.live && ir.sunk)
            RegisterAssign(i, ir);
    
        if(ir.live && ir.exit >= 0)
            AssignSnapshot(dest->exits[ir.exit].snapshot);
    }

    // now the not sunk ones
    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        if(ir.live && !ir.sunk)
            RegisterAssign(i, ir);
    }

}

void JIT::MarkLiveness(IRRef i, IR ir) {
    switch(ir.op) 
    {
#define CASE(Name, ...) case TraceOpCode::Name:
        case TraceOpCode::encode:
        case TraceOpCode::alength:
        case TraceOpCode::glength:
        case TraceOpCode::rep:
        case TraceOpCode::seq:
        case TraceOpCode::gather:
        case TraceOpCode::load:
        case TraceOpCode::reshape:
        case TraceOpCode::phi: 
        BINARY_BYTECODES(CASE)
            {
                code[ir.a].live = true;
                code[ir.b].live = true;
            } break;
        case TraceOpCode::box:
        case TraceOpCode::unbox:
        case TraceOpCode::decodevl:
        case TraceOpCode::decodena:
        case TraceOpCode::length:
        case TraceOpCode::brcast:
        case TraceOpCode::olength:
        case TraceOpCode::lenv:
        case TraceOpCode::denv:
        case TraceOpCode::cenv:
        case TraceOpCode::gproto: 
        case TraceOpCode::gtrue: 
        case TraceOpCode::gfalse: 
            UNARY_FOLD_SCAN_BYTECODES(CASE)
            {
                code[ir.a].live = true;
            } break;
        case TraceOpCode::store:
        case TraceOpCode::newenv:
        case TraceOpCode::scatter: 
        TERNARY_BYTECODES(CASE)
            {
                code[ir.a].live = true;
                code[ir.b].live = true;
                code[ir.c].live = true;
            } break;
        case TraceOpCode::sstore:
            {
                code[ir.c].live = true;
            } break;
        case TraceOpCode::loop:
        case TraceOpCode::jmp:
        case TraceOpCode::exit:
        case TraceOpCode::nest:
        case TraceOpCode::pop:
        case TraceOpCode::curenv:
        case TraceOpCode::nop:
        case TraceOpCode::sload:
        case TraceOpCode::constant:
            // do nothing
            break;
        default: {
                     _error("Unknown op in MarkLiveness");
                 } break;
#undef CASE
    }

    code[ir.in.length].live = true;
    code[ir.out.length].live = true;
    if(ir.exit >= 0)
        MarkSnapshot(dest->exits[ir.exit].snapshot);
}

void JIT::MarkSnapshot(Snapshot const& snapshot) {
    for(size_t i = 0; i < snapshot.stack.size(); i++) {
        code[snapshot.stack[i].environment].live = true;
        code[snapshot.stack[i].env].live = true;
    }

    for(std::map< int64_t, IRRef >::const_iterator i = snapshot.slotValues.begin();
            i != snapshot.slotValues.end(); ++i) {
        code[i->second].live = true;
    }
    
    for(std::set< IRRef >::const_iterator i = snapshot.memory.begin();
            i != snapshot.memory.end(); ++i) {
        code[*i].live = true;
    }
}

bool JIT::AlwaysLive(IR const& ir) {
    return(     ir.op == TraceOpCode::sstore
            ||  ir.op == TraceOpCode::store
            ||  ir.op == TraceOpCode::gtrue
            ||  ir.op == TraceOpCode::gfalse
            ||  ir.op == TraceOpCode::gproto
            ||  ir.op == TraceOpCode::glength
            ||  ir.op == TraceOpCode::push
            ||  ir.op == TraceOpCode::pop 
            ||  ir.op == TraceOpCode::jmp
            ||  ir.op == TraceOpCode::loop
            ||  ir.op == TraceOpCode::exit
            ||  ir.op == TraceOpCode::nest );
}

void JIT::Liveness() {
    // clear all liveness 
    for(size_t i = 0; i < code.size(); i++) {
        code[i].live = false;
    } 

    // traverse loop body
    for(size_t i = code.size()-1; i > Loop; --i) {
        IR& ir = code[i];

        if(AlwaysLive(ir))
            ir.live = true;

        if(ir.live)
            MarkLiveness(i, ir);
    }

    // traverse entire trace & pick up phis
    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        if(AlwaysLive(ir))
            ir.live = true;

        if(ir.op == TraceOpCode::phi && code[ir.a].live )
            ir.live = true;

        if(ir.live)
            MarkLiveness(i, ir);
    }
}
