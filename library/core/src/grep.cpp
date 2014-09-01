
#include <tre/tre.h>
#include <pcre.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
void regex_finalize(Value v) {
    Externalptr const& p = (Externalptr const&)v;
    regex_t* r = (regex_t*)p.ptr();
    delete r;
}

extern "C"
Value regex_compile(State& state, Value const* args) {
    Character pattern = (Character const&)args[0];
    Logical ignorecase = (Logical const&)args[1];
    Logical fixed = (Logical const&)args[2];

    int flags = REG_EXTENDED;
    if(Logical::isTrue(ignorecase[0]))
        flags |= REG_ICASE;
    if(Logical::isTrue(fixed[0]))
        flags |= REG_LITERAL;
    
    regex_t* r = new regex_t();

    int err = tre_regcomp(r, pattern[0]->s, flags);
    
    if(err) {
        printf("Failed to compile regex pattern (%d): %s\n",
            err, pattern[0]->s);
    }

    Value v;
    Externalptr::Init(v, r, Value::Nil(), Value::Nil(), regex_finalize);
    return v;
}

extern "C"
void grep_map(State& state, Logical::Element& s,
        Value const& regex, String text) {
    
    Externalptr const& p = (Externalptr const&)regex;
    regex_t* r = (regex_t*)p.ptr();

    s = (tre_regexec(r, text->s, 0, NULL, 0) == 0)
            ? Logical::TrueElement 
            : Logical::FalseElement;
}

extern "C"
void regex_map(State& state, Integer::Element& s, Integer::Element& l,
        Value const& regex, String text) {
    
    Externalptr const& p = (Externalptr const&)regex;
    regex_t* r = (regex_t*)p.ptr();

    regmatch_t m;
    int match = tre_regexec(r, text->s, 1, &m, 0);
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
void gregex_map(State& state,
        Value& start, Value& length, Value const& regex, String text) {
    
    Externalptr const& p = (Externalptr const&)regex;
    regex_t* r = (regex_t*)p.ptr();

    std::vector<Integer::Element> ss;
    std::vector<Integer::Element> ll;

    regmatch_t m;
    size_t offset = 0;
    int match = 0;
    do {
        match = tre_regexec(r, text->s+offset, 1, &m, 0);
        if(match == 0) {
            ss.push_back(m.rm_so+1+offset);
            ll.push_back(m.rm_eo - m.rm_so);
            offset += m.rm_eo;
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

static std::string replaceAll( 
    std::string const& s,
    std::string const& pattern,
    std::string const& sub )
{
    std::string r;
    std::string::const_iterator current = s.begin();
    std::string::const_iterator next =
        std::search( current, s.end(), pattern.begin(), pattern.end() );
    while( next != s.end() ) {
        r.append( current, next );
        r.append( sub );
        current = next + pattern.size();
        next = std::search( current, s.end(), pattern.begin(), pattern.end() );
    }
    r.append( current, next );
    return r;
}

extern "C"
void sub_map(State& state, Character::Element& out,
        Value const& regex, String text, String sub) {
    
    Externalptr const& p = (Externalptr const&)regex;
    regex_t* r = (regex_t*)p.ptr();

    regmatch_t m[10];
    int match = tre_regexec(r, text->s, 10, m, 0);
    if(match != 0) {
        out = text;
    }
    else {
        std::string s(sub->s);
        // replace all back references in sub
        char c = '1';
        for(int i = 1; i <= 9; ++i, ++c) {
            if(m[i].rm_so >= 0) {
                s = replaceAll(s, std::string("\\")+c, 
                    std::string(text->s+m[i].rm_so, text->s+m[i].rm_eo));
            }
            else {
                s = replaceAll(s, std::string("\\")+c, 
                    std::string(""));
            }
        }
        std::string result;
        result.append( text->s, text->s+m[0].rm_so );
        result.append( s );
        result.append( text->s+m[0].rm_eo, text->s+strlen(text->s) );
        out = state.internStr(result.c_str());
    }
}

extern "C"
void gsub_map(State& state, Character::Element& out,
        Value const& regex, String text, String sub) {
    
    Externalptr const& p = (Externalptr const&)regex;
    regex_t* r = (regex_t*)p.ptr();

    regmatch_t m[10];
    
    size_t offset = 0;
    int match = 0;
    std::string result="";
    do {
        match = tre_regexec(r, text->s+offset, 10, m, 0);
        if(match == 0) {
            std::string s(sub->s);
            // replace all back references in sub
            char c = '1';
            for(int i = 1; i <= 9; ++i, ++c) {
                if(m[i].rm_so >= 0) {
                    s = replaceAll(s, std::string("\\")+c, 
                        std::string(text->s+m[i].rm_so+offset, 
                                    text->s+m[i].rm_eo+offset));
                }
                else {
                    s = replaceAll(s, std::string("\\")+c, 
                        std::string(""));
                }
            }
            result.append( text->s+offset, text->s+m[0].rm_so+offset );
            result.append( s );
            offset += m[0].rm_eo;
        }
    } while( match == 0 );
    out = state.internStr(result.c_str());
}

extern "C"
void agrep_map(State& state, Logical::Element& s,
        Value const& regex, String text, Integer const& costs, Double const& bounds, Integer const& length) {

    Externalptr const& p = (Externalptr const&)regex;
    regex_t* r = (regex_t*)p.ptr();

    regamatch_t match;
    match.nmatch = 0;
    match.pmatch = NULL;

    regaparams_t params;
    params.cost_ins = (int)costs[0];
    params.cost_del = (int)costs[1];
    params.cost_subst = (int)costs[2];

    int64_t cost = std::max(costs[0], std::max(costs[1], costs[2]));

    params.max_cost = Double::isNA(bounds[0])
            ? std::numeric_limits<int>::max()
            : (int)ceil(bounds[0] <= 1 ? bounds[0]*cost*length[0] : bounds[0]);
    params.max_ins = Double::isNA(bounds[0])
            ? std::numeric_limits<int>::max()
            : (int)ceil(bounds[0] <= 1 ? bounds[0]*length[0] : bounds[0]);
    params.max_del = Double::isNA(bounds[0])
            ? std::numeric_limits<int>::max()
            : (int)ceil(bounds[0] <= 1 ? bounds[0]*length[0] : bounds[0]);
    params.max_subst = Double::isNA(bounds[0])
            ? std::numeric_limits<int>::max()
            : (int)ceil(bounds[0] <= 1 ? bounds[0]*length[0] : bounds[0]);
    params.max_err = Double::isNA(bounds[0])
            ? std::numeric_limits<int>::max()
            : (int)ceil(bounds[0] <= 1 ? bounds[0]*length[0] : bounds[0]);

    s = (tre_regaexec(r, text->s, &match, params, 0) == 0)
            ? Logical::TrueElement
            : Logical::FalseElement;
}


extern "C"
void pcre_finalize(Value v) {
    Externalptr const& p = (Externalptr const&)v;
    pcre* r = (pcre*)p.ptr();
    pcre_free(r);
}

extern "C"
Value pcre_regex_compile(State& state, Value const* args) {
    Character pattern = (Character const&)args[0];
    Logical ignorecase = (Logical const&)args[1];

    int flags = 0;
    if(Logical::isTrue(ignorecase[0]))
        flags |= PCRE_CASELESS;
   
    char const* error;
    int erroffset;
    pcre* r = pcre_compile(pattern[0]->s, flags, &error, &erroffset, NULL);

    if(!r)
        printf("PCRE compilation failed at offset %d: %s\n", erroffset, error);

    Value v;
    Externalptr::Init(v, r, Value::Nil(), Value::Nil(), pcre_finalize);
    return v;
}

extern "C"
void pcre_grep_map(State& state, Logical::Element& s,
        Value const& regex, String text) {
    
    Externalptr const& p = (Externalptr const&)regex;
    pcre* r = (pcre*)p.ptr();

    int rc = pcre_exec(r, NULL, text->s, strlen(text->s), 0, 0, 0, 0);

    if(rc < 0 && rc != PCRE_ERROR_NOMATCH)
        printf("PCRE matching error %d\n", rc);

    s = rc >= 0 ? Logical::TrueElement : Logical::FalseElement;
}

extern "C"
void pcre_regex_map(State& state, Integer::Element& s, Integer::Element& l,
        Value const& regex, String text) {
    
    Externalptr const& p = (Externalptr const&)regex;
    pcre* r = (pcre*)p.ptr();

    int ovector[6] = {0};
    int rc = pcre_exec(r, NULL, text->s, strlen(text->s), 0, 0, ovector, 6);
    
    if(rc < 0 && rc != PCRE_ERROR_NOMATCH)
        printf("PCRE matching error %d\n", rc);

    if(rc < 0) {
        s = -1;
        l = -1;
    }
    else {
        s = ovector[0]+1;
        l = ovector[1]-ovector[0];
    }
}

extern "C"
void pcre_gregex_map(State& state,
        Value& start, Value& length, Value const& regex, String text) {
    
    Externalptr const& p = (Externalptr const&)regex;
    pcre* r = (pcre*)p.ptr();

    std::vector<Integer::Element> ss;
    std::vector<Integer::Element> ll;

    size_t len = strlen(text->s);
    size_t offset = 0;
    int rc = 0;
    do {
        int ovector[6] = {0};
        rc = pcre_exec(r, NULL, text->s, len, offset, 0, ovector, 6);
        if(rc < 0 && rc != PCRE_ERROR_NOMATCH)
            printf("PCRE matching error %d\n", rc);

        if( rc >= 0 ) {
            ss.push_back(ovector[0]+1);
            ll.push_back(ovector[1]-ovector[0]);
            offset = ovector[1];
        }
    } while( rc >= 0 );

    Integer s(ss.size());
    Integer l(ll.size());
    for(size_t i = 0; i < ss.size(); ++i) {
        s[i] = ss[i];
        l[i] = ll[i];
    }
    
    start = s;
    length = l;
}

extern "C"
void pcre_sub_map(State& state, Character::Element& out,
        Value const& regex, String text, String sub) {
    
    Externalptr const& p = (Externalptr const&)regex;
    pcre* r = (pcre*)p.ptr();

    int ovector[30] = {0};
    int rc = pcre_exec(r, NULL, text->s, strlen(text->s), 0, 0, ovector, 30);
    
    if(rc < 0 && rc != PCRE_ERROR_NOMATCH)
        printf("PCRE matching error %d\n", rc);

    if(rc < 0) {
        out = text;
    }
    else {
        std::string s(sub->s);
        // replace all back references in sub
        char c = '1';
        for(int i = 1; i <= 9; ++i, ++c) {
            if(i < rc) {
                s = replaceAll(s, std::string("\\")+c, 
                    std::string(text->s+ovector[2*i], text->s+ovector[2*i+1]));
            }
            else {
                s = replaceAll(s, std::string("\\")+c, 
                    std::string(""));
            }
        }
        std::string result;
        result.append( text->s, text->s+ovector[0] );
        result.append( s );
        result.append( text->s+ovector[1], text->s+strlen(text->s) );
        out = state.internStr(result.c_str());
    }
}

extern "C"
void pcre_gsub_map(State& state, Character::Element& out,
        Value const& regex, String text, String sub) {
   
    Externalptr const& p = (Externalptr const&)regex;
    pcre* r = (pcre*)p.ptr();

    size_t len = strlen(text->s);
    size_t offset = 0;
    int rc = 0;
    std::string result="";
    do {
        int ovector[30] = {0};
        rc = pcre_exec(r, NULL, text->s, len, offset, 0, ovector, 30);
        if(rc < 0 && rc != PCRE_ERROR_NOMATCH)
            printf("PCRE matching error %d\n", rc);

        if( rc >= 0 ) {
            std::string s(sub->s);
            // replace all back references in sub
            char c = '1';
            for(int i = 1; i <= 9; ++i, ++c) {
                if(i < rc) {
                    s = replaceAll(s, std::string("\\")+c, 
                        std::string(text->s+ovector[2*i]+offset, 
                                    text->s+ovector[2*i+1]+offset));
                }
                else {
                    s = replaceAll(s, std::string("\\")+c, 
                        std::string(""));
                }
            }
            result.append( text->s+offset, text->s+ovector[0]+offset );
            result.append( s );
            offset += ovector[1];
        }
    } while( rc >= 0 );

    out = state.internStr(result.c_str());
}

