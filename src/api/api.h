
#ifndef API_H
#define API_H

#include <iostream>

#include "../value.h"
#include "../interpreter.h"
#include "../frontend.h"

#define _NYI(M) do { std::cerr << "NYI: " << M << std::endl; throw; } while(0)

struct ScalarString : public Value {
    static const Type::Enum  ValueType = Type::ScalarString;
    static ScalarString& Init(Value& f, String s) {
        Value::Init(f,Type::ScalarString,0);
        f.s = s;
        return (ScalarString&)f;
    }

    String string() const { return s; }
};

VECTOR_IMPL(Integer32, int32_t, false)

    constexpr static const int32_t NAelement = std::numeric_limits<int32_t>::min();


    static bool isNA(int32_t c) { return c == NAelement; }
    static Integer toInteger(Integer32 const& i);
    static Integer32 fromInteger(Integer const& i);
};

VECTOR_IMPL(Logical32, int32_t, false)

    constexpr static const int32_t NAelement = std::numeric_limits<int32_t>::min();

    static bool isNA(int32_t c) { return c == NAelement; }
    static Logical toLogical(Logical32 const& i);
    static Logical32 fromLogical(Logical const& i);
};

// While Riposte treats a pairlist as a normal list internally,
// the R API requires pairlists be actual pairlists where
// references to internal elements can mutate the overall list
// so provide support for that here...
struct Pairlist : public Object {
    static const Type::Enum ValueType = Type::Pairlist;

    struct Inner : public GrayHeapObject {
        SEXP car;
        SEXP cdr;
        String tag;

        Inner(SEXP car, SEXP cdr, String tag)
            : GrayHeapObject(1), car(car), cdr(cdr), tag(tag) {}

        void visit() const;
    };

    static Pairlist& Init(Value& v, SEXP car, SEXP cdr, String tag) {
        Value::Init(v, Type::Pairlist, 0);
        v.p = new Inner(car, cdr, tag);
        return (Pairlist&)v;
    };

    SEXP car() const {
        return ((Inner*)p)->car;
    }
    SEXP cdr() const {
        return ((Inner*)p)->cdr;
    }
    String tag() const {
        return ((Inner*)p)->tag;
    }
    
    void car(SEXP x) const {
        ((Inner*)p)->car = x;
        ((Inner*)p)->writeBarrier();
    }
    void cdr(SEXP x) const {
        ((Inner*)p)->cdr = x;
        ((Inner*)p)->writeBarrier();
    }
    void tag(String x) const {
        ((Inner*)p)->tag = x;
        ((Inner*)p)->writeBarrier();
    }

    static List toList(Pairlist const& pairlist);
    static Value fromList(List const& list);
};

Value ToRiposteValue(Value const& v);

// R API requires interned symbols
// Riposte does not, so translate here...
SEXP ToSEXP(Value const& v);
SEXP ToSEXP(char const* s);

#endif

