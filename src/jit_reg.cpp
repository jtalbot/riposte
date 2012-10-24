#include "jit.h"

void JIT::AssignRegister(size_t index) {
    IR& ir = code[index];
    if(ir.reg <= 0) {

        if(ir.sunk) {
            RegisterAssign(index, ir);
        }

        Register r = { ir.type, ir.out }; 

        if( ir.type == Type::Nil ) return;

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
        //printf("Releasing register %d at %d\n", ir.reg, index);
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

void Assign(JIT* jit, JIT::IRRef a) {
    jit->AssignRegister(a); 
}

void Assign(JIT* jit, JIT::IRRef a, JIT::IRRef b) {
    if(jit->code[a].reg < 0)
        jit->AssignRegister(a);
    if(jit->code[b].reg < 0)
        jit->AssignRegister(b);

    jit->AssignRegister(a); 
    jit->AssignRegister(b); 
}

void Assign(JIT* jit, JIT::IRRef a, JIT::IRRef b, JIT::IRRef c) {
    if(jit->code[a].reg < 0)
        jit->AssignRegister(a);
    if(jit->code[b].reg < 0)
        jit->AssignRegister(b);
    if(jit->code[c].reg < 0)
        jit->AssignRegister(c);

    jit->AssignRegister(a); 
    jit->AssignRegister(b); 
    jit->AssignRegister(c); 
}

void JIT::RegisterAssign(IRRef i, IR ir) {

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
        case TraceOpCode::scatter1: 
        case TraceOpCode::scatter: 
            {
                PreferRegister(ir.a, i);
                Assign(this, ir.a, ir.b, ir.c);
                // necessary to size input
                AssignRegister(code[ir.a].out.length);
            } break;
        case TraceOpCode::store:
        case TraceOpCode::newenv:
            TERNARY_BYTECODES(CASE)
            {
                Assign(this, ir.a, ir.b, ir.c);
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
                Assign(this, ir.a, ir.b);
            } break;
        case TraceOpCode::attrget:
        case TraceOpCode::glength:
        case TraceOpCode::gvalue:
        case TraceOpCode::encode:
        case TraceOpCode::alength:
        case TraceOpCode::rep:
        case TraceOpCode::seq:
        case TraceOpCode::gather1:
        case TraceOpCode::gather:
            BINARY_BYTECODES(CASE)
            {
                Assign(this, ir.a, ir.b);
            } break;
        case TraceOpCode::strip:
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
        case TraceOpCode::random:
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
    /*for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        if( ir.sunk && ir.op != TraceOpCode::unbox &&
            ir.op != TraceOpCode::constant)
        {
            ReleaseRegister(i);
        }

        if(ir.live && (ir.sunk || ir.op == TraceOpCode::phi))
            RegisterAssign(i, ir);

        if(ir.live && ir.exit >= 0)
            AssignSnapshot(dest->exits[ir.exit].snapshot);
    }*/

    // Assign non-sunk registers. Sunk registers 

    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        if(ir.op == TraceOpCode::scatter || ir.op == TraceOpCode::scatter1) {
            // scatter is special since it can resize its register.
            // so resize the register here and then attempt to reuse...
            registers[ir.reg].shape = code[ir.a].out;
        }

        if( !ir.sunk && ir.op != TraceOpCode::unbox &&
            ir.op != TraceOpCode::constant)
        {
            ReleaseRegister(i);
        }

        if(ir.live && !ir.sunk)
            RegisterAssign(i, ir);
       
        // AssignSnapshot traverses the sunk registers
        // Never release registers used in sunk instructions 
        if(ir.live && ir.exit >= 0)
            AssignSnapshot(dest->exits[ir.exit].snapshot);
    }

}

void JIT::Mark(IRRef i, IRRef use) {
    code[i].live = true;
    code[i].use = std::max(code[i].use, use);
}

void JIT::MarkLiveness(IRRef i, IR ir) {
    switch(ir.op) 
    {
#define CASE(Name, ...) case TraceOpCode::Name:
        case TraceOpCode::attrget:
        case TraceOpCode::encode:
        case TraceOpCode::alength:
        case TraceOpCode::glength:
        case TraceOpCode::gvalue:
        case TraceOpCode::rep:
        case TraceOpCode::seq:
        case TraceOpCode::gather1:
        case TraceOpCode::gather:
        case TraceOpCode::load:
        case TraceOpCode::reshape:
        case TraceOpCode::phi: 
        BINARY_BYTECODES(CASE)
            {
                Mark(ir.a, i);
                Mark(ir.b, i);
            } break;
        case TraceOpCode::strip:
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
                Mark(ir.a, i);
            } break;
        case TraceOpCode::store:
        case TraceOpCode::newenv:
        case TraceOpCode::scatter: 
        case TraceOpCode::scatter1: 
        TERNARY_BYTECODES(CASE)
            {
                Mark(ir.a, i);
                Mark(ir.b, i);
                Mark(ir.c, i);
            } break;
        case TraceOpCode::sstore:
            {
                Mark(ir.c, i);
            } break;
        case TraceOpCode::random:
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

    Mark(ir.in.length, i);
    Mark(ir.out.length, i);
    
    if(ir.exit >= 0)
        MarkSnapshot(i, dest->exits[ir.exit].snapshot);
}

void JIT::MarkSnapshot(IRRef ir, Snapshot const& snapshot) {
    for(size_t i = 0; i < snapshot.stack.size(); i++) {
        Mark(snapshot.stack[i].environment, ir);
        Mark(snapshot.stack[i].env, ir);
    }

    for(std::map< int64_t, IRRef >::const_iterator i = snapshot.slotValues.begin();
            i != snapshot.slotValues.end(); ++i) {
        Mark(i->second, ir);
    }
    
    for(std::set< IRRef >::const_iterator i = snapshot.memory.begin();
            i != snapshot.memory.end(); ++i) {
        Mark(*i, ir);
    }
}

bool JIT::AlwaysLive(IR const& ir) {
    return(     ir.op == TraceOpCode::sstore
            ||  ir.op == TraceOpCode::store
            ||  ir.op == TraceOpCode::gtrue
            ||  ir.op == TraceOpCode::gfalse
            ||  ir.op == TraceOpCode::gproto
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
