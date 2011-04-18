/*
 *	A ragel lexer for R.
 *	In ragel, the lexer drives the parsing process, so this also has the basic parsing functions.
 *	Use this to generate parser.cpp
 *      TODO: Eliminate calls to Symbol(state, ...) which require a map search. Should be a hard-coded value.
 *            Parse escape sequences embedded in strings.
 *            Emit complex NA
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
	
	c_comment := 
		any* :>> '*/'
		@{ fgoto main; };

	main := |*

	# Keywords.
	'NULL' 	{token( TOKEN_NULL_CONST, Null::singleton );};
	'NA' 	{token( TOKEN_NUM_CONST, Logical::NA );};
	'TRUE' 	{token( TOKEN_NUM_CONST, Logical::True );};
	'FALSE'	{token( TOKEN_NUM_CONST, Logical::False );};
	'Inf'	{token( TOKEN_NUM_CONST, Double::Inf);};
	'NaN'	{token( TOKEN_NUM_CONST, Double::NaN);};
	'NA_integer_'	{token( TOKEN_NUM_CONST, Integer::NA);};
	'NA_real_'	{token( TOKEN_NUM_CONST, Double::NA);};
	'NA_character_'	{token( TOKEN_STR_CONST, Character::NA);};
	'NA_complex_'	{token( TOKEN_NUM_CONST);};
	'function'	{token( TOKEN_FUNCTION, Symbol(state, "function") );};
	'while'	{token( TOKEN_WHILE, Symbol(state, "while") );};
	'repeat'	{token( TOKEN_REPEAT, Symbol(state, "repeat") );};
	'for'	{token( TOKEN_FOR, Symbol(state, "for") );};
	'if'	{token( TOKEN_IF, Symbol(state, "if") );};
	'in'	{token( TOKEN_IN, Symbol(state, "in") );};
	'else'	{token( TOKEN_ELSE, Symbol(state, "else") );};
	'next'	{token( TOKEN_NEXT, Symbol(state, "next") );};
	'break'	{token( TOKEN_BREAK, Symbol(state, "break") );};
	
	# Single and double-quoted string literals.
	( "'" ( [^'\\\n] | /\\./ )* "'" ) 
		{token( TOKEN_STR_CONST, Character::c(state, std::string(ts+1, te-ts-2)) );};
	( '"' ( [^"\\\n] | /\\./ )* '"' ) 
		{token( TOKEN_STR_CONST, Character::c(state, std::string(ts+1, te-ts-2)) );};

	# Symbols.
	( ('.'? [a-zA-Z_.]) [a-zA-Z0-9_.]* ) 
		{token( TOKEN_SYMBOL, Symbol(state, std::string(ts, te-ts)) );};
	( '`' ( [^"\\\n] | /\\./ )* '`' ) 
		{token( TOKEN_SYMBOL, Symbol(state, std::string(ts+1, te-ts-2)) );};

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
	'=' {token( TOKEN_EQ_ASSIGN, Symbol(state, "=") );};
	'+' {token( TOKEN_PLUS, Symbol(state, "+") );};
	'-' {token( TOKEN_MINUS, Symbol(state, "-") );};
	'^' {token( TOKEN_POW, Symbol(state, "^") );};
	'/' {token( TOKEN_DIVIDE, Symbol(state, "/") );};
	'*' {token( TOKEN_TIMES, Symbol(state, "*") );};
	'**' {token( TOKEN_POW, Symbol(state, "^") );};
	'~' {token( TOKEN_TILDE, Symbol(state, "~") );};
	'$' {token( TOKEN_DOLLAR, Symbol(state, "$") );};
	'@' {token( TOKEN_AT, Symbol(state, "@") );};
	'!' {token( TOKEN_NOT, Symbol(state, "!") );};
	':' {token( TOKEN_COLON, Symbol(state, ":") );};
	'::' {token( TOKEN_NS_GET, Symbol(state, "::") );};
	':::' {token( TOKEN_NS_GET_INT, Symbol(state, ":::") );};
	'&' {token( TOKEN_AND, Symbol(state, "&") );};
	'|' {token( TOKEN_OR, Symbol(state, "|") );};
	'{' {token( TOKEN_LBRACE, Symbol(state, "{") );};
	'}' {token( TOKEN_RBRACE, Symbol(state, "}") );};
	'(' {token( TOKEN_LPAREN, Symbol(state, "(") );};
	')' {token( TOKEN_RPAREN, Symbol(state, ")") );};
	'[' {token( TOKEN_LBRACKET, Symbol(state, "[") );};
	'[[' {token( TOKEN_LBB, Symbol(state, "[[") );};
	']' {token( TOKEN_RBRACKET, Symbol(state, "]") );};
	']]' {token( TOKEN_RBB, Symbol(state, "]]") );};
	'<' {token( TOKEN_LT, Symbol(state, "<") );};
	'>' {token( TOKEN_GT, Symbol(state, ">") );};
	'<=' {token( TOKEN_LE, Symbol(state, "<=") );};
	'>=' {token( TOKEN_GE, Symbol(state, ">=") );};
	'==' {token( TOKEN_EQ, Symbol(state, "==") );};
	'!=' {token( TOKEN_NE, Symbol(state, "!=") );};
	'&&' {token( TOKEN_AND2, Symbol(state, "&&") );};
	'||' {token( TOKEN_OR2, Symbol(state, "||") );};
	'<-' {token( TOKEN_LEFT_ASSIGN, Symbol(state, "<-") );};
	'->' {token( TOKEN_RIGHT_ASSIGN, Symbol(state, "->") );};
	'->>' {token( TOKEN_RIGHT_ASSIGN, Symbol(state, "->>") );};
	'<<-' {token( TOKEN_LEFT_ASSIGN, Symbol(state, "<<-") );};
	'?' {token( TOKEN_QUESTION, Symbol(state, "?") );};
	
	# Special Operators.
	('%' [^\n]* '%') {token(TOKEN_SPECIALOP, Symbol(state, std::string(ts, te-ts)) ); };

	# Separators.
	',' {token( TOKEN_COMMA );};
	';' {token( TOKEN_SEMICOLON );};
	
	# Comments and whitespace.
	'/*' { fgoto c_comment; };
	'#' [^\n]*;
	'\n' {token( TOKEN_NEWLINE );};
	( any - 33..126 )+;

	*|;
}%%

void Parser::token( int tok, Value v)
{
	Parser::Result result;
	const char *data = ts;
	int len = te - ts;

	/*std::cout << '<' << tok << "> ";
	std::cout.write( data, len );
	std::cout << '\n';*/

	Parse(pParser, tok, v, &result); 
	
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
}

Parser::Parser(State& state) : line(0), col(0), have(0), state(state)
{
	%% write init;
}

int Parser::execute( const char* data, int len, bool isEof, Value& result)
{
	Result r;
	r.state = 0;

	pParser = ParseAlloc(malloc);

	const char *p = data;
	const char *pe = data+len;
	const char* eof = isEof ? pe : 0;

	%% write exec;

	Parse(pParser, 0, Value::NIL, &r);
	ParseFree(pParser, free);

	result = r.value;

	if( cs == Scanner_error || r.state == -1 )
		return -1;
	else if( cs >= Scanner_first_final && r.state == 1)
		return 1;
	else
		return 0;
}

int Parser::buffer_execute( )
{
	/*static char buf[16384];

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
	*/
	return 0;
}

int Parser::finish()
{
	if( cs == Scanner_error )
		return -1;
	else if( cs >= Scanner_first_final )
		return 1;
	else
		return 0;
}

