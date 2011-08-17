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
#include "tokens.h"
#include "ast.cpp"

%%{
	machine Scanner; 
	write data;

	# Numeric literals.
	float = digit* '.' digit+ | digit+ '.'?;
	exponent = [eE] [+\-]? digit+;
	hexponent = [pP] [+\-]? digit+;
	
	main := |*

	# Keywords.
	'NULL' 	{token( TOKEN_NULL_CONST, Null::Singleton() );};
	'NA' 	{token( TOKEN_NUM_CONST, Logical::NA() );};
	'TRUE' 	{token( TOKEN_NUM_CONST, Logical::True() );};
	'FALSE'	{token( TOKEN_NUM_CONST, Logical::False() );};
	'Inf'	{token( TOKEN_NUM_CONST, Double::Inf() );};
	'NaN'	{token( TOKEN_NUM_CONST, Double::NaN() );};
	'NA_integer_'	{token( TOKEN_NUM_CONST, Integer::NA() );};
	'NA_real_'	{token( TOKEN_NUM_CONST, Double::NA() );};
	'NA_character_'	{token( TOKEN_STR_CONST, Character::NA() );};
	'NA_complex_'	{token( TOKEN_NUM_CONST, Complex::NA() );};
	'function'	{token( TOKEN_FUNCTION, Symbol::function );};
	'while'	{token( TOKEN_WHILE, Symbol::whileSym );};
	'repeat'	{token( TOKEN_REPEAT, Symbol::repeatSym );};
	'for'	{token( TOKEN_FOR, Symbol::forSym );};
	'if'	{token( TOKEN_IF, Symbol::ifSym );};
	'in'	{token( TOKEN_IN );};
	'else'	{token( TOKEN_ELSE );};
	'next'	{token( TOKEN_NEXT, Symbol::nextSym );};
	'break'	{token( TOKEN_BREAK, Symbol::breakSym );};
	
	# Single and double-quoted string literals.
	( "'" ( [^'\\\n] | /\\./ )* "'" ) 
		{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(state.StrToSym(unescape(s))) );};
	( '"' ( [^"\\\n] | /\\./ )* '"' ) 
		{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(state.StrToSym(unescape(s))) );};

	# Symbols.
	( (('.' [a-zA-Z_.]) | [a-zA-Z_]) [a-zA-Z0-9_.]* ) 
		{token( TOKEN_SYMBOL, Symbol(state.StrToSym(std::string(ts, te-ts))) );};
	( '`' ( [^`\\\n] | /\\./ )* '`' ) 
		{std::string s(ts+1, te-ts-2); token( TOKEN_SYMBOL, Symbol(state.StrToSym(unescape(s))) );};

	# Numeric literals.
	( float exponent? ) 
		{token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );};
	
	( float exponent? 'i' ) 
		{token( TOKEN_NUM_CONST, Complex::c(0, atof(std::string(ts, te-ts-1).c_str())) );};
	
	( float exponent? 'L' ) 
		{token( TOKEN_NUM_CONST, Integer::c(atof(std::string(ts, te-ts-1).c_str())) );};
	
	# Integer octal. Leading part buffered by float.
	#( '0' [0-9]+ [ulUL]{0,2} ) 
	#	{token( TK_IntegerOctal );};

	# Hex. Leading 0 buffered by float.
	( '0' ( 'x' [0-9a-fA-F]+ hexponent?) ) 
		{token( TOKEN_NUM_CONST );};

	# Operators. 
	'=' {token( TOKEN_EQ_ASSIGN, Symbol::eqassign );};
	'+' {token( TOKEN_PLUS, Symbol::add );};
	'-' {token( TOKEN_MINUS, Symbol::sub );};
	'^' {token( TOKEN_POW, Symbol::pow );};
	'/' {token( TOKEN_DIVIDE, Symbol::div );};
	'*' {token( TOKEN_TIMES, Symbol::mul );};
	'**' {token( TOKEN_POW, Symbol::pow );};
	'~' {token( TOKEN_TILDE, Symbol::tilde );};
	'$' {token( TOKEN_DOLLAR, Symbol::dollar );};
	'@' {token( TOKEN_AT, Symbol::at );};
	'!' {token( TOKEN_NOT, Symbol::lnot );};
	':' {token( TOKEN_COLON, Symbol::colon );};
	'::' {token( TOKEN_NS_GET, Symbol::nsget );};
	':::' {token( TOKEN_NS_GET_INT, Symbol::nsgetint );};
	'&' {token( TOKEN_AND, Symbol::land );};
	'|' {token( TOKEN_OR, Symbol::lor );};
	'{' {token( TOKEN_LBRACE, Symbol::brace );};
	'}' {token( TOKEN_RBRACE );};
	'(' {token( TOKEN_LPAREN, Symbol::paren );};
	')' {token( TOKEN_RPAREN );};
	'[' {token( TOKEN_LBRACKET, Symbol::bracket );};
	'[[' {token( TOKEN_LBB, Symbol::bb );};
	']' {token( TOKEN_RBRACKET );};
	'<' {token( TOKEN_LT, Symbol::lt );};
	'>' {token( TOKEN_GT, Symbol::gt );};
	'<=' {token( TOKEN_LE, Symbol::le );};
	'>=' {token( TOKEN_GE, Symbol::ge );};
	'==' {token( TOKEN_EQ, Symbol::eq );};
	'!=' {token( TOKEN_NE, Symbol::neq );};
	'&&' {token( TOKEN_AND2, Symbol::sland );};
	'||' {token( TOKEN_OR2, Symbol::slor );};
	'<-' {token( TOKEN_LEFT_ASSIGN, Symbol::assign );};
	'->' {token( TOKEN_RIGHT_ASSIGN, Symbol::assign );};
	'->>' {token( TOKEN_RIGHT_ASSIGN, Symbol::assign2 );};
	'<<-' {token( TOKEN_LEFT_ASSIGN, Symbol::assign2 );};
	'?' {token( TOKEN_QUESTION, Symbol::question );};
	
	# Special Operators.
	('%' [^\n%]* '%') {token(TOKEN_SPECIALOP, Symbol(state.StrToSym(std::string(ts, te-ts))) ); };

	# Separators.
	',' {token( TOKEN_COMMA );};
	';' {token( TOKEN_SEMICOLON );};
	# the parser expects to never get two NEWLINE tokens in a row.
	((any-(10 | 33..126))* ('#' [^\n]* '\n' | '\n'))+ {token( TOKEN_NEWLINE );};
	
	# Discard all other characters
	( any - (10 | 33..126) )+;

	*|;
}%%

void Parser::token( int tok, Value v)
{
	const char *data = ts;
	int len = te - ts;

	/*std::cout << '<' << tok << "> ";
	std::cout.write( data, len );
	std::cout << '\n';*/

	int initialErrors = errors;

	// Do the lookahead to resolve the dangling else conflict
	if(lastTokenWasNL) {
		if(tok != TOKEN_ELSE && (nesting.size()==0 || nesting.top()!=TOKEN_LPAREN))
			Parse(pParser, TOKEN_NEWLINE, Nil, this);
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

	if(tok == TOKEN_LPAREN) nesting.push(tok);
	else if(tok == TOKEN_LBRACE) nesting.push(tok);
	else if(tok == TOKEN_RPAREN || tok == TOKEN_RBRACE) nesting.pop();
	else if(tok == TOKEN_FUNCTION) source.push(ts);

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
		std::cout << "Error (" << intToStr(line+1) << "," << intToStr(col+1) << ") : unexpected '" << std::string(data, len) + "'" << std::endl; 
	}
}

Parser::Parser(State& state) : line(0), col(0), state(state), errors(0), complete(false), lastTokenWasNL(false) 
{}

int Parser::execute( const char* data, int len, bool isEof, Value& out, FILE* trace )
{
	out = Nil;
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
	Parse(pParser, 0, Nil, this);
	ParseFree(pParser, free);
	errors = syntaxErrors;

	if( cs == Scanner_error && syntaxErrors == 0) {
		syntaxErrors++;
		std::cout << "Lexing error (" << intToStr(line+1) << "," << intToStr(col+1) << ") : unexpected '" << std::string(ts, te-ts) + "'" << std::endl; 
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
/*
int Parser::buffer_execute( )
{
	static char buf[16384];

	std::ios::sync_with_stdio(false);

	bool done = false;
	while ( !done ) {
		char* b = buf + have;
		const char *p = b;
		int space = 16384 - have;

		if ( space == 0 ) {
			std::cerr << "OUT OF BUFFER SPACE" << std::endl;
			return -1;
		}

		std::cin.read( b, space );
		int len = std::cin.gcount();
		const char *pe = p + len;
		const char *eof = 0;

	 	if ( std::cin.eof() ) {
			eof = pe;
			done = true;
		}

		%% write exec;

		if ( cs == Scanner_error ) {
			std::cerr << "PARSE ERROR" << std::endl;
			return -1;
		}

		if ( ts == 0 )
			have = 0;
		else {
			have = pe - ts;
			memmove( buf, ts, have );
			te -= (ts-buf);
			ts = buf;
		}
	}
	return 0;
}
*/

