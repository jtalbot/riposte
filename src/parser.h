
#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <stdlib.h>
#include <iostream>
#include "value.h"

struct Parser {
	struct Result {
		Value value;
		int state;
	};

	int line, col;
	const char *ts;
	const char *te;
	int act, have;
	int cs;
	State& state;
	void* pParser;

	// R language needs more than 1 lookahead to resolve dangling else
	// if we see a newline, have to wait to send to parser. if next token is an else, discard newline and emit else.
	// otherwise, emit newline and next token
	bool lastTokenWasNL;
	
	void token( int tok, Value v=Value::NIL );

	Parser(State& state); 
	int execute( const char* data, int len, bool isEof, Value& result );
	int buffer_execute();
	int finish();
};


#endif

