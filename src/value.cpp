
#include "value.h"
#include "internal.h"

DEFINE_ENUM_TO_STRING(String, STRINGS)

_doublena doublena = {0x7fff000000001953};

const unsigned char Null::NAelement = 255;

const unsigned char Raw::NAelement = 255;

const unsigned char Logical::NAelement = 255;

const int64_t Integer::NAelement = std::numeric_limits<int64_t>::min();

const double Double::NAelement = doublena.d;

const std::complex<double> Complex::NAelement = std::complex<double>(doublena.d, doublena.d);

const Symbol Character::NAelement = Symbols::NA;

const Value List::NAelement = Null::Singleton();

//const Value PairList::NAelement = Null::Singleton();

//const Value Call::NAelement = Null::Singleton();

//const Value Expression::NAelement = Null::Singleton();

//namespace Symbols {
//#define CONST_DEFN(name, string, ...) const Symbol name(String::name);
//STRINGS(CONST_DEFN)
//#undef CONST_DEFN
//}
