
#include "jit.h"

class GuardPass {

    std::vector<JIT::IR>& code;
    std::vector<Value> const& constants;
    std::map<JIT::IRRef, JIT::IRRef> eq;
    std::vector<JIT::IRRef> forward;

    JIT::IRRef substitute(JIT::IRRef i) {
        JIT::IR& ir = code[i];

        if( ir.op == TraceOpCode::glength ) {
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
            if(len > 16 || mustDespecialize) {
                ir.op = TraceOpCode::length;
            }
            // otherwise, propogate length down
            else {
                ir.op = TraceOpCode::nop;
                return ir.b;
            }
        }
        return i;
    }

    void eval(JIT::IRRef i) {
        JIT::IR const& ir = code[i];
        
        // Union euqal lengths
        if( ir.op == TraceOpCode::eq
         && code[ir.a].op == TraceOpCode::glength
         && code[ir.b].op == TraceOpCode::glength ) 
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
    GuardPass(std::vector<JIT::IR>& code, std::vector<Value> const& constants )
        : code(code), constants(constants) {}

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
                if(code[ir.a].op == TraceOpCode::glength)
                    reshaped.insert(ir.a);
                if(code[ir.b].op == TraceOpCode::glength)
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

void JIT::StrengthenGuards(void) {
    GuardPass pass(code, constants);
    pass.run();
}
