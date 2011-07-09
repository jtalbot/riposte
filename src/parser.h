
#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <stdlib.h>
#include <iostream>
#include <stack>
#include <algorithm>
#include <locale>
#include "value.h"

// trim from end
static inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

struct Parser {

	int line, col;
	State& state;
	void* pParser;
	const char *ts, *te, *le;
	
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

	// To provide user with function source have to track beginning locations
	// Parser pops when function rule is reduced
	std::stack<const char*> source;
	Character popSource() {
		assert(source.size() > 0);
		std::string s(source.top(), le-source.top());
		Character result = Character::c(state.StrToSym(rtrim(s)));
		source.pop();
		return result;
	}

	void token( int tok, Value v=Value::Nil );

	Parser(State& state); 
	int execute( const char* data, int len, bool isEof, Value& result, FILE* trace=NULL );
};


#endif

