
#include "../../../src/runtime.h"
#include "../../../src/interpreter.h"

extern "C"
Value importIntoEnv(State& state, Value const* args) {

    auto out = static_cast<REnvironment const&>(args[0]).environment();
    auto out_names = static_cast<Character const&>(args[1]);

    auto in = static_cast<REnvironment const&>(args[2]).environment();
    auto in_names = static_cast<Character const&>(args[3]);

    assert(out_names.length() == in_names.length());

    for(size_t i = 0; i < out_names.length(); ++i) {
        Value const* v = in->get(in_names[i]);
        out->insert(out_names[i]) = v ? *v : Value::Nil();
    }
    out->writeBarrier();
    
    return args[0]; 
}

