
#include <tre/tre.h>
#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
void regex_finalize(Value v) {
    Externalptr const& p = (Externalptr const&)v;
    regex_t* r = (regex_t*)p.ptr();
    delete r;
}

extern "C"
Value regex_compile(Thread& thread, Value const* args) {
    Character pattern = (Character const&)args[0];
    Logical ignorecase = (Logical const&)args[1];
    Logical fixed = (Logical const&)args[2];

    int flags = REG_EXTENDED;
    if(Logical::isTrue(ignorecase[0]))
        flags |= REG_ICASE;
    if(Logical::isTrue(fixed[0]))
        flags |= REG_LITERAL;
    
    regex_t* r = new regex_t();

    tre_regcomp(r, pattern[0], flags);

    Value v;
    Externalptr::Init(v, r, Value::Nil(), Value::Nil(), regex_finalize);
    return v;
}

extern "C"
void grep_map(Thread& thread, Logical::Element& s,
        Value const& regex, String text) {
    
    Externalptr const& p = (Externalptr const&)regex;
    regex_t* r = (regex_t*)p.ptr();

    s = (tre_regexec(r, text, 0, NULL, 0) == 0)
            ? Logical::TrueElement 
            : Logical::FalseElement;
}

extern "C"
void regex_map(Thread& thread, Integer::Element& s, Integer::Element& l,
        Value const& regex, String text) {
    
    Externalptr const& p = (Externalptr const&)regex;
    regex_t* r = (regex_t*)p.ptr();

    regmatch_t m;
    int match = tre_regexec(r, text, 1, &m, 0);
    if(match != 0) {
        s = -1;
        l = -1;
    }
    else {
        s = m.rm_so+1;
        l = m.rm_eo - m.rm_so;
    }
}

extern "C"
void gregex_map(Thread& thread,
        Value& start, Value& length, Value const& regex, String text) {
    
    Externalptr const& p = (Externalptr const&)regex;
    regex_t* r = (regex_t*)p.ptr();

    std::vector<Integer::Element> ss;
    std::vector<Integer::Element> ll;

    regmatch_t m;
    size_t offset = 0;
    int match = 0;
    do {
        match = tre_regexec(r, text+offset, 1, &m, 0);
        if(match == 0) {
            ss.push_back(m.rm_so+1+offset);
            ll.push_back(m.rm_eo - m.rm_so);
            offset += m.rm_so+1;
        }
    } while( match == 0 );

    Integer s(ss.size());
    Integer l(ll.size());
    for(size_t i = 0; i < ss.size(); ++i) {
        s[i] = ss[i];
        l[i] = ll[i];
    }
    
    start = s;
    length = l;
}

