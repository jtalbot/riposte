
#ifndef _RIPOSTE_SYMBOLS_H
#define _RIPOSTE_SYMBOLS_H

// predefined symbols

#define SYMBOLS_ENUM(_, p) 				\
	_(NA, "<NA>", p) /* this should have the same string representation as something else so it will be masked in the string table. */ \
	_(NArep, "<NA>", p) \
	_(empty, "", p) \
	_(dots, "...", p) \
	_(Logical, "logical", p)\
	_(Integer, "integer", p)\
	_(Numeric, "numeric", p)\
	_(Double, "double", p)\
	_(Complex, "complex", p)\
	_(Character, "character", p)\
	_(Raw, "raw", p)\
	_(internal, ".Internal", p) \
	_(assign, "<-", p) \
	_(assign2, "<<-", p) \
	_(eqassign, "=", p) \
	_(function, "function", p) \
	_(returnSym, "return", p) \
	_(forSym, "for", p) \
	_(whileSym, "while", p) \
	_(repeatSym, "repeat", p) \
	_(nextSym, "next", p) \
	_(breakSym, "break", p) \
	_(ifSym, "if", p) \
	_(brace, "{", p) \
	_(paren, "(", p) \
	_(add, "+", p) \
	_(sub, "-", p) \
	_(mul, "*", p) \
	_(tilde, "~", p) \
	_(div, "/", p) \
	_(idiv, "%/%", p) \
	_(pow, "^", p) \
	_(mod, "%%", p) \
	_(lnot, "!", p) \
	_(land, "&", p) \
	_(sland, "&&", p) \
	_(lor, "|", p) \
	_(slor, "||", p) \
	_(eq, "==", p) \
	_(neq, "!=", p) \
	_(lt, "<", p) \
	_(le, "<=", p) \
	_(gt, ">", p) \
	_(ge, ">=", p) \
	_(dollar, "$", p) \
	_(at, "@", p) \
	_(colon, ":", p) \
	_(nsget, "::", p) \
	_(nsgetint, ":::", p) \
	_(bracket, "[", p) \
	_(bb, "[[", p) \
	_(question, "?", p) \
	_(abs, "abs", p) \
	_(sign, "sign", p) \
	_(sqrt, "sqrt", p) \
	_(floor, "floor", p) \
	_(ceiling, "ceiling", p) \
	_(trunc, "trunc", p) \
	_(round, "round", p) \
	_(signif, "signif", p) \
	_(exp, "exp", p) \
	_(log, "log", p) \
	_(cos, "cos", p) \
	_(sin, "sin", p) \
	_(tan, "tan", p) \
	_(acos, "acos", p) \
	_(asin, "asin", p) \
	_(atan, "atan", p) \
	_(names, "names", p) \
	_(dim, "dim", p) \
	_(classSym, "class", p) \
	_(bracketAssign, "[<-", p) \
	_(bbAssign, "[[<-", p) \
	_(UseMethod, "UseMethod", p) \
	_(seq_len, "seq_len", p) \
	_(TRUE, "TRUE", p) \
	_(FALSE, "FALSE", p) \
	_(type, "typeof", p) \


class SymbolTable {
	std::map<std::string, int64_t> symbolTable;
	std::map<int64_t, std::string> reverseSymbolTable;
	int64_t next;
public:
	SymbolTable();

	int64_t in(std::string const& s) {
		std::map<std::string, int64_t>::const_iterator i = symbolTable.find(s);
		if(i == symbolTable.end()) {
			int64_t index = next++;
			symbolTable[s] = index;
			reverseSymbolTable[index] = s;
			return index;
		} else return i->second;
	
	}

	std::string const& out(int64_t i) const {
		return reverseSymbolTable.find(i)->second;
	}

};

#endif
