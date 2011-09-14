
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

const Value List::NAelement = Value::Nil();

uint64_t Environment::globalRevision = 1;

