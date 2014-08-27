
#ifndef _RIPOSTE_PARSER_H
#define _RIPOSTE_PARSER_H

#include "value.h"

class Global;

int parse(Global& global, char const* filename,
    char const* code, size_t len, bool isEof, Value& result, FILE* trace=NULL);

#endif
