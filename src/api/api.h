
#ifndef API_H
#define API_H

#include <iostream>

#include "../value.h"

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
    static bool isNA(int32_t c) { return c == NAelement; }
    static Integer toInteger(Integer32 const& i);
    static Integer32 fromInteger(Integer const& i);
};

VECTOR_IMPL(Logical32, int32_t, false)
    static bool isNA(int32_t c) { return c == NAelement; }
    static Logical toLogical(Logical32 const& i);
    static Logical32 fromLogical(Logical const& i);
};

Value ToRiposteValue(Value const& v);

#endif

