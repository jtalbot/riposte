
#include "value.h"

_doublena doublena = {0x7fff000000001953};

const unsigned char Null::NAelement = 255;
const unsigned char Null::LowerBound = 255;
const unsigned char Null::UpperBound = 255;

const unsigned char Raw::NAelement = 255;
const unsigned char Raw::LowerBound = 0;
const unsigned char Raw::UpperBound = 255;

const char Logical::TrueElement = -1;
const char Logical::FalseElement = 0;
const char Logical::NAelement = 1;
const char Logical::LowerBound = 0;
const char Logical::UpperBound = -1;

const int64_t Integer::NAelement = std::numeric_limits<int64_t>::min();
const int64_t Integer::LowerBound = std::numeric_limits<int64_t>::min()+1;
const int64_t Integer::UpperBound = std::numeric_limits<int64_t>::max();

const double Double::NAelement = doublena.d;
const int64_t Double::NAelementInt = doublena.i;
const double Double::LowerBound = -std::numeric_limits<double>::infinity();
const double Double::UpperBound = std::numeric_limits<double>::infinity();

const String Character::NAelement = Strings::NA;
const String Character::LowerBound = Strings::empty;
const String Character::UpperBound = Strings::Maximal;

const Value List::NAelement = Value::Nil();
const Value List::LowerBound = Value::Nil();
const Value List::UpperBound = Value::Nil();

