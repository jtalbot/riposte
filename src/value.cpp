
#include "value.h"

const unsigned char Null::NAelement;

const char Logical::TrueElement;
const char Logical::FalseElement;
const char Logical::NAelement;

const int64_t Integer::NAelement;

Double::_doublena doublena = {0x7fff000000001953};

const double Double::NAelement = doublena.d;

const String Character::NAelement;

const unsigned char Raw::NAelement;

const Value List::NAelement;

