#include "strings.h"

#define DEFINE(name, string, ...) String Strings::name = string;
STRINGS(DEFINE)
#undef DEFINE

String Strings::pos = Strings::add;
String Strings::neg = Strings::sub;
