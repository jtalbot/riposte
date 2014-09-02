/*
 *	A ragel lexer for R.
 *	In ragel, the lexer drives the parsing process, so this also has the basic parsing functions.
 *	Use this to generate parser.cpp
 *      TODO: 
 *            Parse escape sequences embedded in strings.
 *            Include the double-to-int warnings, e.g. on 1.0L  and 1.5L
 *            Generate hex numbers
 *            Do we really want to allow '.' inside hex numbers? R allows them, but ignores them when parsing to a number.
 *            R parser has a rule for OP % OP. Is this ever used?
 */

#include "parser.h"
#include "../interpreter.h"
#include "grammar.h"
#include "grammar.cpp"

%%{
	machine Scanner; 
	write data;

	# Numeric literals.
	float = digit* '.' digit+ | digit+ '.'?;
	exponent = [eE] [+\-]? digit+;
	hexponent = [pP] [+\-]? digit+;
	
	main := |*

	# Keywords.
    'Nil'   {token( TOKEN_NIL_CONST );};
	'NULL' 	{token( TOKEN_NULL_CONST, Null::Singleton() );};
	'NA' 	{token( TOKEN_NUM_CONST, Logical::NA() );};
	'TRUE' 	{token( TOKEN_NUM_CONST, Logical::True() );};
	'FALSE'	{token( TOKEN_NUM_CONST, Logical::False() );};
	'Inf'	{token( TOKEN_NUM_CONST, Double::Inf() );};
	'NaN'	{token( TOKEN_NUM_CONST, Double::NaN() );};
	'NA_integer_'	{token( TOKEN_NUM_CONST, Integer::NA() );};
	'NA_real_'	{token( TOKEN_NUM_CONST, Double::NA() );};
	'NA_character_'	{token( TOKEN_STR_CONST, Character::NA() );};
	# 'NA_complex_'	{token( TOKEN_NUM_CONST, Complex::NA() );};
	'NA_list_'	{token( TOKEN_STR_CONST, List::NA() );};
	'function'	{token( TOKEN_FUNCTION, CreateSymbol(Strings::function) );};
	'while'	{token( TOKEN_WHILE, CreateSymbol(Strings::whileSym) );};
	'repeat'	{token( TOKEN_REPEAT, CreateSymbol(Strings::repeatSym) );};
	'for'	{token( TOKEN_FOR, CreateSymbol(Strings::forSym) );};
	'if'	{token( TOKEN_IF, CreateSymbol(Strings::ifSym) );};
	'in'	{token( TOKEN_IN );};
	'else'	{token( TOKEN_ELSE );};
	'next'	{token( TOKEN_NEXT, CreateSymbol(Strings::nextSym) );};
	'break'	{token( TOKEN_BREAK, CreateSymbol(Strings::breakSym) );};
	
	# Single and double-quoted string literals.
	( "'" ( [^'\\] | /\\./ )* "'" ) 
		{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(global.internStr(unescape(s))) );};
	( '"' ( [^"\\] | /\\./ )* '"' ) 
		{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(global.internStr(unescape(s))) );};

	# CreateSymbols.
	( '..' digit+ )
		{token( TOKEN_SYMBOL, CreateSymbol(global.internStr(std::string(ts, te-ts))));};

	( ('.' ([a-zA-Z_.] [a-zA-Z0-9_.]*)?) | [a-zA-Z] [a-zA-Z0-9_.]* ) 
		{token( TOKEN_SYMBOL, CreateSymbol(global.internStr(std::string(ts, te-ts))) );};
	( '`' ( [^`\\] | /\\./ )* '`' ) 
		{std::string s(ts+1, te-ts-2); token( TOKEN_SYMBOL, CreateSymbol(global.internStr(unescape(s))) );};
	# Numeric literals.
	( float exponent? ) 
		{token( TOKEN_NUM_CONST, Double::c(strtod(std::string(ts, te).c_str(), NULL)) );};
	
	( float exponent? 'L' ) 
		{token( TOKEN_NUM_CONST, Integer::c(strtoll(std::string(ts, te-1).c_str(), NULL, 10)) );};
	
	( float exponent? 'i' ) 
		{token( TOKEN_NUM_CONST, CreateComplex(strtod(std::string(ts, te-1).c_str(), NULL)) );};
	
	# Integer octal. Leading part buffered by float.
	#( '0' [0-9]+ [ulUL]{0,2} ) 
	#	{token( TK_IntegerOctal );};

	# Hex. Leading 0 buffered by float.
	( '0' ( 'x' [0-9a-fA-F]+ hexponent?) ) 
		{token( TOKEN_NUM_CONST, Double::c(hexStrToInt(std::string(ts,te-ts))) );};

	( '0' ( 'x' [0-9a-fA-F]+ hexponent? 'L') ) 
		{token( TOKEN_NUM_CONST, Integer::c(hexStrToInt(std::string(ts,te-ts))) );};
	
    # Operators. 
	'=' {token( TOKEN_EQ_ASSIGN, CreateSymbol(Strings::eqassign) );};
	'+' {token( TOKEN_PLUS, CreateSymbol(Strings::add) );};
	'-' {token( TOKEN_MINUS, CreateSymbol(Strings::sub) );};
	'^' {token( TOKEN_POW, CreateSymbol(Strings::pow) );};
	'/' {token( TOKEN_DIVIDE, CreateSymbol(Strings::div) );};
	'*' {token( TOKEN_TIMES, CreateSymbol(Strings::mul) );};
	'**' {token( TOKEN_POW, CreateSymbol(Strings::pow) );};
	'~' {token( TOKEN_TILDE, CreateSymbol(Strings::tilde) );};
	'$' {token( TOKEN_DOLLAR, CreateSymbol(Strings::dollar) );};
	'@' {token( TOKEN_AT, CreateSymbol(Strings::at) );};
	'!' {token( TOKEN_NOT, CreateSymbol(Strings::lnot) );};
	':' {token( TOKEN_COLON, CreateSymbol(Strings::colon) );};
	'::' {token( TOKEN_NS_GET, CreateSymbol(Strings::nsget) );};
	':::' {token( TOKEN_NS_GET_INT, CreateSymbol(Strings::nsgetint) );};
	'&' {token( TOKEN_AND, CreateSymbol(Strings::land) );};
	'|' {token( TOKEN_OR, CreateSymbol(Strings::lor) );};
	'{' {token( TOKEN_LBRACE, CreateSymbol(Strings::brace) );};
	'}' {token( TOKEN_RBRACE );};
	'(' {token( TOKEN_LPAREN, CreateSymbol(Strings::paren) );};
	')' {token( TOKEN_RPAREN );};
	'[' {token( TOKEN_LBRACKET, CreateSymbol(Strings::bracket) );};
	'[[' {token( TOKEN_LBB, CreateSymbol(Strings::bb) );};
	']' {token( TOKEN_RBRACKET );};
	'<' {token( TOKEN_LT, CreateSymbol(Strings::lt) );};
	'>' {token( TOKEN_GT, CreateSymbol(Strings::gt) );};
	'<=' {token( TOKEN_LE, CreateSymbol(Strings::le) );};
	'>=' {token( TOKEN_GE, CreateSymbol(Strings::ge) );};
	'==' {token( TOKEN_EQ, CreateSymbol(Strings::eq) );};
	'!=' {token( TOKEN_NE, CreateSymbol(Strings::neq) );};
	'&&' {token( TOKEN_AND2, CreateSymbol(Strings::land2) );};
	'||' {token( TOKEN_OR2, CreateSymbol(Strings::lor2) );};
	'<-' {token( TOKEN_LEFT_ASSIGN, CreateSymbol(Strings::assign) );};
	'->' {token( TOKEN_RIGHT_ASSIGN, CreateSymbol(Strings::assign) );};
	'->>' {token( TOKEN_RIGHT_ASSIGN, CreateSymbol(Strings::assign2) );};
	'<<-' {token( TOKEN_LEFT_ASSIGN, CreateSymbol(Strings::assign2) );};
	'?' {token( TOKEN_QUESTION, CreateSymbol(Strings::question) );};
	
	# Special Operators.
	('%' [^\n%]* '%') {token(TOKEN_SPECIALOP, CreateSymbol(global.internStr(std::string(ts, te-ts))) ); };

	# Separators.
	',' {token( TOKEN_COMMA );};
	';' {token( TOKEN_SEMICOLON );};
	# the parser expects to never get two NEWLINE tokens in a row.
	((any-(10 | 33..126))* ('#' [^\n]* '\n' | '\n'))+ {token( TOKEN_NEWLINE );};
	
	# Discard all other characters
	( any - (10 | 33..126) )+;

	*|;
}%%

void Parser::token(int tok, Value v)
{
	const char *data = ts;
	int len = te - ts;

	/*std::cout << '<' << tok << "> ";
	std::cout.write( data, len );
	std::cout << '\n';*/

	int initialErrors = errors;

	// Do the lookahead to resolve the dangling else conflict
	if(lastTokenWasNL) {
		if(tok != TOKEN_ELSE && 
            (nesting.size()==0 ||
                (nesting.top()!=TOKEN_LPAREN &&
                 nesting.top()!=TOKEN_LBRACKET &&
                 nesting.top()!=TOKEN_LBB)))
			Parse(pParser, TOKEN_NEWLINE, Value::Nil(), this);
		Parse(pParser, tok, v, this);
		lastTokenWasNL = false;
	}
	else {
		if(tok == TOKEN_NEWLINE)
			lastTokenWasNL = true;
		else
			Parse(pParser, tok, v, this);
	}

	le = te;

	if( tok == TOKEN_LPAREN ||
	    tok == TOKEN_LBRACKET ||
	    tok == TOKEN_LBB || 
	    tok == TOKEN_LBRACE)
        nesting.push(tok);
	else if(
        tok == TOKEN_RPAREN || 
        tok == TOKEN_RBRACE)
        nesting.pop();
    else if(
        tok == TOKEN_RBRACKET) {
        // need to do a bit of extra work to catch ]] vs. ]
        if(nesting.top() == TOKEN_LBRACKET) {
            nesting.pop();
        }
        else {
            nesting.pop();
            nesting.push(TOKEN_LBRACKET);
        }
    }
	else if(tok == TOKEN_FUNCTION)
        source.push(ts);

	/* Count newlines and columns. Use for error reporting? */ 
	for ( int i = 0; i < len; i ++ ) {
		if ( data[i] == '\n' ) {
			line += 1;
			col = 1;
		}
		else {
			col += 1;
		}
	}

	if(errors > initialErrors) {
		std::cout << "Error (" << filename << ":" << intToStr(line+1) << "," 
            << intToStr(col+1) << ") : unexpected '" 
            << std::string(data, len) + "'" << std::endl; 
	}
}

Parser::Parser(Global& global, char const* filename) : line(0), col(0), global(global), filename(filename), errors(0), complete(false), lastTokenWasNL(false) 
{}

int Parser::execute( const char* data, int len, bool isEof, Value& out, FILE* trace )
{
	out = Value::Nil();
	errors = 0;
	lastTokenWasNL = false;
	complete = false;

	pParser = ParseAlloc(malloc);

	/*ParseTrace(trace, 0);*/

	const char *p = data;
	const char *pe = data+len;
	const char* eof = isEof ? pe : 0;
	int cs, act;
	%% write init;
	%% write exec;
	int syntaxErrors = errors;
	Parse(pParser, 0, Value::Nil(), this);
	ParseFree(pParser, free);
	errors = syntaxErrors;

	if( cs == Scanner_error && syntaxErrors == 0) {
		syntaxErrors++;
		std::cout << "Lexing error (" << filename << ":" << intToStr(line+1) << ":" << intToStr(col+1) << ")" << std::endl;
        // TODO: make a meaningful error. te can be 0x0 sometimes! Try entering `z
        // (" << intToStr(line+1) << "," << intToStr(col+1) << ") : unexpected '" << std::string(ts, te-ts) + "'" << std::endl; 
	}
	
	if( syntaxErrors > 0 )
		return -1;
	else if( cs >= Scanner_first_final && complete) {
		out = result;
		return 1;
	} 
	else
		return 0;
}

String Parser::popSource() {
	assert(source.size() > 0);
	std::string s(source.top(), le-source.top());
	String result = global.internStr(rtrim(s));
	source.pop();
	return result;	
}

int parse(Global& global, char const* filename,
    char const* code, size_t len, bool isEof, Value& result, FILE* trace) {
    Parser parser(global, filename);
    return parser.execute(code, len, isEof, result, trace);
}

