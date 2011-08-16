
#ifndef _RIPOSTE_SYMBOLS_H
#define _RIPOSTE_SYMBOLS_H

// predefined strings

#define STRINGS(_) 				\
	_(NA, "<NA>") /* this should have the same string representation as something else so it will be masked in the string table. */ \
	_(NArep, "<NA>") \
	_(empty, "") \
	_(dots, "...") \
	_(Logical, "logical")\
	_(Integer, "integer")\
	_(Numeric, "numeric")\
	_(Double, "double")\
	_(Complex, "complex")\
	_(Character, "character")\
	_(Name, "name") \
	_(Raw, "raw")\
	_(internal, ".Internal") \
	_(assign, "<-") \
	_(assign2, "<<-") \
	_(eqassign, "=") \
	_(function, "function") \
	_(returnSym, "return") \
	_(forSym, "for") \
	_(whileSym, "while") \
	_(repeatSym, "repeat") \
	_(nextSym, "next") \
	_(breakSym, "break") \
	_(ifSym, "if") \
	_(brace, "{") \
	_(paren, "(") \
	_(add, "+") \
	_(sub, "-") \
	_(mul, "*") \
	_(tilde, "~") \
	_(div, "/") \
	_(idiv, "%/%") \
	_(pow, "^") \
	_(mod, "%%") \
	_(lnot, "!") \
	_(land, "&") \
	_(sland, "&&") \
	_(lor, "|") \
	_(slor, "||") \
	_(eq, "==") \
	_(neq, "!=") \
	_(lt, "<") \
	_(le, "<=") \
	_(gt, ">") \
	_(ge, ">=") \
	_(dollar, "$") \
	_(at, "@") \
	_(colon, ":") \
	_(nsget, "::") \
	_(nsgetint, ":::") \
	_(bracket, "[") \
	_(bb, "[[") \
	_(question, "?") \
	_(abs, "abs") \
	_(sign, "sign") \
	_(sqrt, "sqrt") \
	_(floor, "floor") \
	_(ceiling, "ceiling") \
	_(trunc, "trunc") \
	_(round, "round") \
	_(signif, "signif") \
	_(exp, "exp") \
	_(log, "log") \
	_(cos, "cos") \
	_(sin, "sin") \
	_(tan, "tan") \
	_(acos, "acos") \
	_(asin, "asin") \
	_(atan, "atan") \
	_(names, "names") \
	_(dim, "dim") \
	_(classSym, "class") \
	_(bracketAssign, "[<-") \
	_(bbAssign, "[[<-") \
	_(UseMethod, "UseMethod") \
	_(seq_len, "seq_len") \
	_(TRUE, "TRUE") \
	_(FALSE, "FALSE") \
	_(type, "typeof") \
	_(value, "value") \
	_(dotGeneric, ".Generic") \
	_(dotMethod, ".Method") \
	_(dotClass, ".Class") \

DECLARE_ENUM(String, STRINGS)

class StringTable {
	std::map<std::string, int64_t> stringTable;
	std::map<int64_t, std::string> reverseStringTable;
	int64_t next;
public:
	StringTable() : next(0) {
		// insert predefined strings into table at known positions (corresponding to their enum value)
	#define ENUM_STRING_TABLE(name, string) \
		stringTable[string] = String::name; \
		reverseStringTable[String::name] = string;\
		assert(next==String::name);\
		next++;\

		STRINGS(ENUM_STRING_TABLE);
	}



	int64_t in(std::string const& s) {
		std::map<std::string, int64_t>::const_iterator i = stringTable.find(s);
		if(i == stringTable.end()) {
			int64_t index = next++;
			stringTable[s] = index;
			reverseStringTable[index] = s;
			return index;
		} else return i->second;
	
	}

	std::string const& out(int64_t i) const {
		return reverseStringTable.find(i)->second;
	}

};

#endif
