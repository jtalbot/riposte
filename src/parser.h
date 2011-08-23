
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
static inline std::string& rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

static inline std::string& unescape(std::string& s) {
	std::string::const_iterator i = s.begin();

	// first do fast check for any escape sequences...
	bool escaped = false;
	while(i != s.end())
		if(*i++ == '\\') escaped = true;

	if(!escaped) return s;

	std::string r;
	i = s.begin();
	while(i != s.end())
	{
		char c = *i++;
		if(c == '\\' && i != s.end())
		{
			switch(*i++) {
				case 'a': r += '\a'; break;
				case 'b': r += '\b'; break;
				case 'f': r += '\f'; break;
				case 'n': r += '\n'; break;
				case 'r': r += '\r'; break;
				case 't': r += '\t'; break;
				case 'v': r += '\v'; break;
				case '\\': r += '\\'; break;
				case '"': r += '"'; break;
				case '\'': r += '\''; break;
				case ' ': r += ' '; break;
				case '\n': r += '\n'; break;
				default: throw RiposteError(std::string("Unrecognized escape in \"") + s + "\""); break;
			}
		}
		else r += c;
	}
	s = r;
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
	Symbol popSource() {
		assert(source.size() > 0);
		std::string s(source.top(), le-source.top());
		Symbol result = state.StrToSym(rtrim(s));
		source.pop();
		return result;
	}

	void token( int tok, Value v=Value::Nil() );

	Parser(State& state); 
	int execute( const char* data, int len, bool isEof, Value& result, FILE* trace=NULL );
};

struct Pairs {
	struct Pair {
		Symbol n;
		Value v;        
	};        
	std::deque<Pair, traceable_allocator<Value> > p;
	int64_t length() const { return p.size(); }        
	void push_front(Symbol n, Value const& v) { Pair t = {n, v}; p.push_front(t); } 
	void push_back(Symbol n, Value const& v)  { Pair t = {n, v}; p.push_back(t); }        
	const Value& value(int64_t i) const { return p[i].v; }
	const Symbol& name(int64_t i) const { return p[i].n; }

	List values() const {
		List l(length());
		for(int64_t i = 0; i < length(); i++)
			l[i] = value(i);
		return l;
	}

	Value names(bool forceNames) const {
		bool named = false;
		for(int64_t i = 0; i < length(); i++) {                        
			if(name(i) != Symbols::empty) {
				named = true;
				break;                        
			}                
		}                
		if(named || forceNames) {
			Character n(length());                        
			for(int64_t i = 0; i < length(); i++)
				n[i] = Symbol(name(i).i);
			return n;
		}
		else return Value::Nil();
	}
};



#endif

