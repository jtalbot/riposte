#include "string.h"

#define DEFINE(name, string, ...) String Strings::name = string;
STRINGS(DEFINE)
#undef DEFINE
