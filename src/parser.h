
#ifndef _RIPOSTE_PARSER_H
#define _RIPOSTE_PARSER_H

#include "value.h"

int parse(State& state, char const* filename,
    char const* code, size_t len, bool isEof, Value& result, FILE* trace=NULL);

#endif
