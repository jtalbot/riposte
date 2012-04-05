
#include "value.h"

_doublena doublena = {0x7fff000000001953};

const uint8_t Null::NAelement = 255;

const uint8_t Raw::NAelement = 255;

const int8_t Logical::TrueElement = -1;
const int8_t Logical::FalseElement = 0;
const int8_t Logical::NAelement = 1;

const int64_t Integer::NAelement = std::numeric_limits<int64_t>::min();

const double Double::NAelement = doublena.d;

const String Character::NAelement = Strings::NA;

const Value List::NAelement = Value::Nil();

