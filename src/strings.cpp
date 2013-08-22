#include "strings.h"

// Strings are actually initialized in State::State

#define DEFINE(name, string, ...) String Strings::name = 0;
STRINGS(DEFINE)
#undef DEFINE

