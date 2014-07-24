
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
void dyn_finalize(Value v) {
    Externalptr const& p = (Externalptr const&)v;
    dlclose(p.ptr());
}

extern "C"
Value dynload(Thread& thread, Value const* args) {
    Character const& name = (Character const&)args[0];
    Logical const& local = (Logical const&)args[1];
    Logical const& now = (Logical const&)args[2];

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
Value dynunload(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    dlclose(p.ptr());
    return Value::Nil();
}

extern "C"
Value dynsym(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    Character const& name = (Character const&)args[1];
    
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
        return Null::Singleton();
    }
}

