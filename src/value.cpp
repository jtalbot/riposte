
#include "value.h"
#include "internal.h"

_doublena doublena = {0x7fff000000001953};

const bool Null::CheckNA = false;
const unsigned char Null::NAelement = 255;

const bool Raw::CheckNA = false;
const unsigned char Raw::NAelement = 255;

const bool Logical::CheckNA = true;
const unsigned char Logical::NAelement = 255;

const bool Integer::CheckNA = true;
const int64_t Integer::NAelement = std::numeric_limits<int64_t>::min();

const bool Double::CheckNA = false;
const double Double::NAelement = doublena.d;

const bool Complex::CheckNA = false;
const std::complex<double> Complex::NAelement = std::complex<double>(doublena.d, doublena.d);

const bool Character::CheckNA = true;
const Symbol Character::NAelement = Symbol::NA;

const bool List::CheckNA = false;
const Value List::NAelement = Null::Singleton();

const bool PairList::CheckNA = false;
const Value PairList::NAelement = Null::Singleton();

const bool Call::CheckNA = false;
const Value Call::NAelement = Null::Singleton();

const bool Expression::CheckNA = false;
const Value Expression::NAelement = Null::Singleton();

#define CONST_DEFN(name, string, ...) const Symbol Symbol::name(String::name);
STRINGS(CONST_DEFN)
#undef CONST_DEFN

