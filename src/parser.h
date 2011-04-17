
#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <stdlib.h>
#include <iostream>
#include "value.h"

struct Parser {
	int line, col;
	const char *ts;
	const char *te;
	int act, have;
	int cs;
	State& state;
	void* pParser;
	
	void token( int tok, Value v=Value::NIL );

	Parser(State& state); 
	int execute( const char* data, int len, bool isEof, Value& result );
	int buffer_execute();
	int finish();
};


#endif

