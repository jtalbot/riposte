
#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <stdlib.h>
#include <iostream>
#include <stack>
#include "value.h"

struct Parser {

	int line, col;
	State& state;
	void* pParser;
	const char *ts, * te;
	
	Value result;
	int errors;
	bool complete;

	// R language needs more than 1 lookahead to resolve dangling else
	// if we see a newline, have to wait to send to parser. if next token is an else, discard newline and emit else.
	// otherwise, emit newline and next token
	bool lastTokenWasNL;

	// If we're inside parentheses, we should discard all newlines
	// If we're in the top level or in curly braces, we have to preserve newlines
	std::stack<int> nesting;

	void token( int tok, Value v=Value::NIL );

	Parser(State& state); 
	int execute( const char* data, int len, bool isEof, Value& result, FILE* trace=NULL );
};


#endif

