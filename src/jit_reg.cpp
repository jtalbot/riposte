#include "jit.h"

void JIT::AssignRegister(size_t index) {
    IR& ir = code[index];
    ir.live = true;
    if(ir.reg <= 0) {
        Register r = { ir.type, ir.out }; 

        if( ir.op == TraceOpCode::constant 
                || ir.op == TraceOpCode::load
                || ir.op == TraceOpCode::sload ) {
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

void JIT::AssignSnapshot(Snapshot const& snapshot) {
    for(size_t i = 0; i < snapshot.stack.size(); i++) {
        AssignRegister(snapshot.stack[i].environment);
        AssignRegister(snapshot.stack[i].env);
    }

    for(std::map< int64_t, IRRef >::const_iterator i = snapshot.slotValues.begin();
            i != snapshot.slotValues.end(); ++i) {
        AssignRegister(i->second);
    }
    
    for(std::map< int64_t, IRRef >::const_iterator i = snapshot.slotLengths.begin();
            i != snapshot.slotLengths.end(); ++i) {
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

    if(ir.op != TraceOpCode::load && ir.op != TraceOpCode::constant)
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
                AssignSnapshot(exits[i].snapshot);
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
                AssignSnapshot(exits[i].snapshot);
            } break;
        case TraceOpCode::elength:
        case TraceOpCode::alength:
        case TraceOpCode::rep:
        case TraceOpCode::seq:
        case TraceOpCode::gather:
            BINARY_BYTECODES(CASE)
            {
                AssignRegister(std::max(ir.a, ir.b));
                AssignRegister(std::min(ir.a, ir.b));
            } break;
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


    // This should work:
    //  Backwards pass over loop body
    //  Mark stores as live and propogate.
    //  At loop header, traverse phis and mark (a) live
    //  Traverse loop header

 
    // add all register to the freeRegisters list
    for(size_t i = 0; i < registers.size(); i++) {
        freeRegisters.insert( std::make_pair(registers[i], i) );
    }

    // clear all liveness flags
    for(size_t i = 0; i < code.size(); i++) {
        code[i].live = false;
        code[i].reg = 0;
    } 

    // traverse loop body
    for(size_t i = code.size()-1; i > Loop; --i) {
        IR& ir = code[i];

        if(     ir.op == TraceOpCode::sstore
            ||  ir.op == TraceOpCode::store
            ||  ir.op == TraceOpCode::gtrue
            ||  ir.op == TraceOpCode::gfalse
            ||  ir.op == TraceOpCode::gproto
            ||  ir.op == TraceOpCode::push
            ||  ir.op == TraceOpCode::pop )
            ir.live = true;

        if(ir.live)
            RegisterAssign(i, ir);
    }

    // retraverse to pick up phis
    registers.clear();
    registers.push_back(invalid);
    freeRegisters.clear();

    // add all register to the freeRegisters list
    for(size_t i = 0; i < registers.size(); i++) {
        freeRegisters.insert( std::make_pair(registers[i], i) );
    }

    // clear all liveness flags
    for(size_t i = 0; i < code.size(); i++) {
        code[i].reg = 0;
    } 

    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        if(     ir.op == TraceOpCode::sstore
            ||  ir.op == TraceOpCode::store
            ||  ir.op == TraceOpCode::gtrue
            ||  ir.op == TraceOpCode::gfalse
            ||  ir.op == TraceOpCode::gproto
            ||  ir.op == TraceOpCode::push
            ||  ir.op == TraceOpCode::pop 
            ||  ir.op == TraceOpCode::jmp
            ||  ir.op == TraceOpCode::loop
            ||  ir.op == TraceOpCode::exit
            ||  ir.op == TraceOpCode::nest )
            ir.live = true;

        if(     ir.op == TraceOpCode::phi &&
                code[ir.a].live )
            ir.live = true;

        if(ir.live && ir.sunk)
            RegisterAssign(i, ir);
    }

    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        if(     ir.op == TraceOpCode::sstore
            ||  ir.op == TraceOpCode::store
            ||  ir.op == TraceOpCode::gtrue
            ||  ir.op == TraceOpCode::gfalse
            ||  ir.op == TraceOpCode::gproto
            ||  ir.op == TraceOpCode::push
            ||  ir.op == TraceOpCode::pop 
            ||  ir.op == TraceOpCode::jmp
            ||  ir.op == TraceOpCode::loop
            ||  ir.op == TraceOpCode::exit
            ||  ir.op == TraceOpCode::nest )
            ir.live = true;

        if(     ir.op == TraceOpCode::phi &&
                code[ir.a].live )
            ir.live = true;

        if(ir.live && !ir.sunk)
            RegisterAssign(i, ir);
    }

}

