
#include "jit.h"

class GuardPass {

    std::vector<JIT::IR>& code;
    std::vector<Value> const& constants;
    std::map<JIT::IRRef, JIT::IRRef> eq;
    std::vector<JIT::IRRef> forward;
    JIT::IRRef Loop;

    int64_t specializationLength;

    JIT::IRRef substitute(JIT::IRRef i) {
        JIT::IR& ir = code[i];

        JIT::IRRef f = dofind(i);
        if(f != i) {
            return f;
        }
        return i;
    }

    JIT::IRRef specialize(JIT::IRRef i, std::set<JIT::IRRef> const& variant) {
        
        JIT::IRRef j = substitute(i);
        
        JIT::IR& ir = code[i];
      
        if( ir.op == TraceOpCode::length
         || ir.op == TraceOpCode::rlength
         || ir.op == TraceOpCode::resize ) {

            int64_t olen = observedLength(ir.c);
            if( olen > 0 
             && olen <= specializationLength
             && variant.find(j) == variant.end() ) {
                dounion(i, ir.c);
                return ir.c;
            }

            if(j != i) {
               ir.c = j;
            }
            else {
                ir.exit = -1;
            }
        } 
        
        return j; 
    }

    int64_t observedLength(JIT::IRRef i) {
        if(code[i].op == TraceOpCode::constant) {
            return ((Integer const&)constants[code[i].a])[0];
        }
        else {
            return ((Integer const&)constants[code[i].c])[0];
        }
    }

    void eval(JIT::IRRef i, std::set<JIT::IRRef> const& variant) {
        JIT::IR const& ir = code[i];
      
        if( ir.op == TraceOpCode::rlength ) {
            int64_t alen = observedLength(ir.a);
            int64_t blen = observedLength(ir.b);
            int64_t olen = observedLength(ir.c);
            
            if(alen == blen && code[ir.a].op != TraceOpCode::constant && code[ir.b].op != TraceOpCode::constant) {
                if( (variant.find(ir.a) != variant.end())
                 == (variant.find(i) != variant.end()) )
                    dounion(ir.a, i);
                if( (variant.find(ir.b) != variant.end())
                 == (variant.find(i) != variant.end()) )
                    dounion(ir.b, i);
            }
            else if(alen == olen && alen % blen == 0 && code[ir.a].op != TraceOpCode::constant) {
                if( (variant.find(ir.a) != variant.end())
                 == (variant.find(i) != variant.end()) )
                    dounion(ir.a, i);
            }
            else if(blen == olen && blen % alen == 0 && code[ir.b].op != TraceOpCode::constant) {
                if( (variant.find(ir.a) != variant.end())
                 == (variant.find(i) != variant.end()) )
                    dounion(ir.b, i);
            }
        }

        if( ir.op == TraceOpCode::resize ) {
            int64_t alen = observedLength(ir.a);
            int64_t olen = observedLength(ir.c);
           
            /*if(alen == olen) {
                dounion(ir.a, i);
            }*/
        }

        if( ir.op == TraceOpCode::geq ) {
            if( (code[ir.a].op == TraceOpCode::length
              || code[ir.a].op == TraceOpCode::rlength
              || code[ir.a].op == TraceOpCode::resize)
             && code[ir.b].op == TraceOpCode::constant )
                dounion(ir.a, ir.b);
            if( (code[ir.b].op == TraceOpCode::length
              || code[ir.b].op == TraceOpCode::rlength
              || code[ir.b].op == TraceOpCode::resize)
             && code[ir.a].op == TraceOpCode::constant )
                dounion(ir.a, ir.b);
        }

    }

    void dounion(JIT::IRRef a, JIT::IRRef b) {
        //printf("Unioning %d %d\n", a, b);
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
    GuardPass(std::vector<JIT::IR>& code, std::vector<Value> const& constants, int64_t specializationLength, JIT::IRRef Loop)
        : code(code), constants(constants), specializationLength(specializationLength), Loop(Loop) {}

    void varies(std::set<JIT::IRRef>& variant, JIT::IRRef a, JIT::IRRef s) {
        if(code[a].op != TraceOpCode::constant) {
            //printf("Variant: %d from %d\n", a, s);
            variant.insert(a);
        }
    }

    void run() {

        // mark non-loop invariant lengths
        std::set<JIT::IRRef> variant;
        for(JIT::IRRef j = code.size()-1; j < code.size(); --j) {
            JIT::IR const& ir = code[j];
            /*if(ir.op == TraceOpCode::phi &&
                    ir.a != ir.b && 
                    (code[ir.a].op == TraceOpCode::length ||
                     code[ir.a].op == TraceOpCode::rlength ||
                     code[ir.a].op == TraceOpCode::resize)) {
                varies(variant, ir.a, j);
                varies(variant, ir.b, j);
            }
            
            if(ir.op == TraceOpCode::length && j > Loop)
                varies(variant, j, j);

            if(ir.op == TraceOpCode::resize &&
               (  variant.find(j) != variant.end() ) )
                varies(variant, ir.a, j);

            if(ir.op == TraceOpCode::rlength &&
               (  variant.find(j) != variant.end() ) ) {
                int64_t alen = observedLength(ir.a);
                int64_t blen = observedLength(ir.b);
                int64_t olen = observedLength(ir.c);
                if(alen == olen)
                    varies(variant, ir.a, j);
                if(blen == olen)
                    varies(variant, ir.b, j);
            }*/

            if(ir.op == TraceOpCode::resize)
                varies(variant, ir.a, j);
        }

        // find lengths that could be profitably specialized to be equal
        std::set<JIT::IRRef> resized;
        for(JIT::IRRef j = 0; j < code.size(); ++j) {
            if(specializationLength >= 0)
                eval(j, variant);
        }

        // substitute equal lengths and do concrete specialization
        std::vector<JIT::IRRef> forward(code.size(),0);
        forward[0] = 0;
        forward[1] = 1;
        forward[2] = 2;
        forward[3] = 3;


        for(JIT::IRRef j = 0; j < code.size(); ++j) {

            JIT::IR& ir = code[j];
            ir = JIT::Forward(ir, forward);

            //forward[j] = substitute(j);

            // mark all reshaped lengths as not specializable.
            /*if(ir.op == TraceOpCode::reshape) {
              if(code[ir.a].op == TraceOpCode::glength || code[ir.a].op == TraceOpCode::gvalue)
              reshaped.insert(ir.a);
              if(code[ir.b].op == TraceOpCode::glength || code[ir.b].op == TraceOpCode::gvalue)
              reshaped.insert(ir.b);
              }*/

            forward[j] = specialize(j, variant);
        }

        // despecialize
        /*for(JIT::IRRef j = 0; j < code.size(); ++j) {
          JIT::IR& ir = code[j];
          ir = JIT::Forward(ir, forward);
          forward[j] = despecialize(j, reshaped.find(j) != reshaped.end());
          }*/
    }
};

void JIT::StrengthenGuards(int64_t specializationLength) {
    GuardPass pass(code, constants, specializationLength, Loop);
    pass.run();
}
