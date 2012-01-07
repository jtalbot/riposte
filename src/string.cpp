#include "string.h"

#define DEFINE(name, string, ...) String Strings::name = String::Init(string);
STRINGS(DEFINE)
#undef DEFINE
