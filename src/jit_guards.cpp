
#include "jit.h"

class GuardPass {

    std::vector<JIT::IR>& code;
    std::vector<Value> const& constants;
    std::map<JIT::IRRef, JIT::IRRef> eq;
    std::vector<JIT::IRRef> forward;

    size_t specializationLength;

    JIT::IRRef substitute(JIT::IRRef i) {
        JIT::IR& ir = code[i];

        if( (ir.op == TraceOpCode::glength ||
             ir.op == TraceOpCode::gvalue)) {
            JIT::IRRef f = dofind(i);
            if(f != i) {
                ir.b = f;
                return f;
            }
        }
        return i;
    }

    JIT::IRRef despecialize(JIT::IRRef i, bool mustDespecialize) {
        JIT::IR& ir = code[i];
        
        if( ir.op == TraceOpCode::glength && code[ir.b].op == TraceOpCode::constant  ) {
            int64_t len = ((Integer const&)constants[code[ir.b].a])[0];

            // if too long, despecialize
            if(len > specializationLength || mustDespecialize) {
                ir.op = TraceOpCode::length;
            }
            // otherwise, propogate length down
            else {
                ir.op = TraceOpCode::nop;
                return ir.b;
            }
        }
        
        else if( ir.op == TraceOpCode::gvalue && code[ir.b].op == TraceOpCode::constant  ) {
            int64_t len = ((Integer const&)constants[code[ir.b].a])[0];

            ir.op = TraceOpCode::nop;

            // if too long, despecialize
            if(len > specializationLength || mustDespecialize) {
                return ir.a;
            }
            // otherwise, propogate length down
            else {
                return ir.b;
            }
        }
        return i;
    }

    void eval(JIT::IRRef i) {
        JIT::IR const& ir = code[i];
       
        // Union equal lengths
        if( ir.op == TraceOpCode::eq
         && (code[ir.a].op == TraceOpCode::glength || code[ir.a].op == TraceOpCode::gvalue || code[ir.a].op == TraceOpCode::length )
         && (code[ir.b].op == TraceOpCode::glength || code[ir.b].op == TraceOpCode::gvalue || code[ir.a].op == TraceOpCode::length ) ) 
        {
            dounion(ir.a, ir.b);
        }
    }

    void dounion(JIT::IRRef a, JIT::IRRef b) {

        JIT::IRRef ar = dofind(a);
        JIT::IRRef br = dofind(b);

        if(ar == br)
            return;

        eq[ std::max(ar, br) ] = std::min(ar, br);
    }

    JIT::IRRef dofind(JIT::IRRef a) {
        std::map<JIT::IRRef, JIT::IRRef>::const_iterator i = eq.find(a);
        if(i != eq.end()) {
            JIT::IRRef p = dofind(i->second);
            if(p != i->second)
                eq[a] = p;
            return p;
        }
        else {
            return a;
        }
    }


public:
    GuardPass(std::vector<JIT::IR>& code, std::vector<Value> const& constants, size_t specializationLength)
        : code(code), constants(constants), specializationLength(specializationLength) {}

    void run() {
        // find lengths that are equal
        for(JIT::IRRef j = 0; j < code.size(); ++j) {
            eval(j);
        }

        // substitute equal lengths and despecialize 
        std::vector<JIT::IRRef> forward(code.size(),0);
        forward[0] = 0;
        forward[1] = 1;
        forward[2] = 2;
        forward[3] = 3;

        std::set<JIT::IRRef> reshaped;

        for(JIT::IRRef j = 0; j < code.size(); ++j) {
            
            JIT::IR& ir = code[j];
            ir = JIT::Forward(ir, forward);

            forward[j] = substitute(j);
        
            // mark all reshaped lengths as not specializable.
            if(ir.op == TraceOpCode::reshape) {
                if(code[ir.a].op == TraceOpCode::glength || code[ir.a].op == TraceOpCode::gvalue)
                    reshaped.insert(ir.a);
                if(code[ir.b].op == TraceOpCode::glength || code[ir.b].op == TraceOpCode::gvalue)
                    reshaped.insert(ir.b);
            }
        }

        // despecialize
        for(JIT::IRRef j = 0; j < code.size(); ++j) {
            JIT::IR& ir = code[j];
            ir = JIT::Forward(ir, forward);
            forward[j] = despecialize(j, reshaped.find(j) != reshaped.end());
        }
    }
};

void JIT::StrengthenGuards(size_t specializationLength) {
    GuardPass pass(code, constants, specializationLength);
    pass.run();
}
