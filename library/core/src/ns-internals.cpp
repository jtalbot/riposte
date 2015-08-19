
#include "../../../src/runtime.h"
#include "../../../src/interpreter.h"

extern "C"
Value importIntoEnv(State& state, Value const* args) {

    Environment* out = ((REnvironment const&)args[0]).environment();    
    Character const& out_names = (Character const&)args[1];

    Environment const* in = ((REnvironment const&)args[2]).environment();    
    Character const& in_names = (Character const&)args[3];

    assert(out_names.length() == in_names.length());

    for(size_t i = 0; i < out_names.length(); ++i) {
        Value const* v = in->get(in_names[i]);
        out->insert(out_names[i]) = v ? *v : Value::Nil();
    }
    
    return args[0]; 
}

