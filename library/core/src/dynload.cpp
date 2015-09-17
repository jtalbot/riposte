
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
void dyn_finalize(Value v) {
    auto p = static_cast<Externalptr const&>(v);
    dlclose(p.ptr());
}

extern "C"
Value dynload(State& state, Value const* args) {
    auto name = static_cast<Character const&>(args[0]);
    auto local = static_cast<Logical const&>(args[1]);
    auto now = static_cast<Logical const&>(args[2]);

    int f1 = Logical::isTrue(local[0]) ? RTLD_LOCAL : RTLD_GLOBAL;
    int f2 = Logical::isTrue(now[0]) ? RTLD_NOW : RTLD_LAZY;

    void* p = dlopen(name.s->s, f1|f2);

    if(!p) {
        std::cerr << dlerror() << std::endl;
    }

    Value v;
    Externalptr::Init(v, p, Value::Nil(), Value::Nil(), dyn_finalize);
    return v;
}

extern "C"
Value dynunload(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    dlclose(p.ptr());
    return Value::Nil();
}

extern "C"
Value dynsym(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    auto name = static_cast<Character const&>(args[1]);
    
    dlerror();
    void* r = dlsym(p.ptr(), name.s->s);

    char* err = dlerror();
    if(err) {
        std::cerr << err << std::endl;
    }

    if(r) {
        Value v;
        Externalptr::Init(v, r, Value::Nil(), Value::Nil(), NULL);
        return v;
    }
    else {
        return Null();
    }
}

