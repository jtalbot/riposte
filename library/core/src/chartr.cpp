
#include <string>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
void chartr_finalize(Value v) {
    Externalptr const& p = (Externalptr const&)v;
    unsigned char* r = (unsigned char*)p.ptr();
    delete [] r;
}

extern "C"
Value chartr_compile(Thread& thread, Value const* args) {
    Character o = (Character const&)args[0];
    Character n = (Character const&)args[1];
    unsigned char* map = new unsigned char[256];
    for(size_t i = 0; i < 256; i++) map[i] = i;
    size_t i = 0;
    while(o.s[i]) {
        map[(unsigned char)o.s[i]] = n.s[i];
        ++i;
    }

    Value v;
    Externalptr::Init(v, map, Value::Nil(), Value::Nil(), chartr_finalize);
    return v;
}

extern "C"
void chartr_map(Thread& thread,
        Character::Element& result,
        Character::Element text,
        Value const& map) {
    Externalptr const& p = (Externalptr const&)map;
    unsigned char* m = (unsigned char*)p.ptr();

    std::string t(text);
    for(size_t i = 0; i < t.size(); i++) {
        t[i] = m[(unsigned char)t[i]];
    }
    result = thread.internStr(t.c_str());
}
