
#ifndef _RIPOSTE_SYMBOLS_H
#define _RIPOSTE_SYMBOLS_H

#include <tbb/tbb.h>
#include "enum.h"

// predefined strings

#define STRINGS(_) 			\
	/* must match types */ 		\
	_(Null, 	"NULL")		\
	_(Raw, 		"raw")		\
	_(Logical, 	"logical")	\
	_(Integer, 	"integer")	\
	_(Double, 	"double")	\
	_(Complex, 	"complex")	\
	_(Character, 	"character")	\
	_(List,		"list")		\
	_(Function,	"function")	\
	_(BuiltIn,	"builtin")	\
	_(Environment,	"environment")	\
	_(Object,	"object")	\
	_(Promise, 	"promise") 	\
	_(Nil,		"nil")		\
	/* Now all other strings */	\
	_(NA, 		"<NA>") /* this should have the same string representation as something else so it will be masked in the string table. */ \
	_(NArep, 	"<NA>") \
	_(empty, 	"") \
	_(dots, 	"...") \
	_(Name, 	"name") \
	_(Numeric, 	"numeric") \
	_(PairList,	"pairlist")	\
	_(Symbol,	"symbol")	\
	_(Call,		"call")		\
	_(Expression,	"expression")	\
	_(internal, 	".Internal") \
	_(assign, 	"<-") \
	_(assign2, 	"<<-") \
	_(eqassign, 	"=") \
	_(function, 	"function") \
	_(returnSym, 	"return") \
	_(forSym, 	"for") \
	_(whileSym, 	"while") \
	_(repeatSym, 	"repeat") \
	_(nextSym, 	"next") \
	_(breakSym,	"break") \
	_(ifSym, 	"if") \
	_(switchSym, 	"switch") \
	_(brace, 	"{") \
	_(paren, 	"(") \
	_(add, 		"+") \
	_(sub, 		"-") \
	_(mul, 		"*") \
	_(tilde, 	"~") \
	_(div, 		"/") \
	_(idiv, 	"%/%") \
	_(pow, 		"^") \
	_(mod, 		"%%") \
	_(atan2, 	"atan2") \
	_(hypot, 	"hypot") \
	_(lnot, 	"!") \
	_(land, 	"&") \
	_(sland, 	"&&") \
	_(lor, 		"|") \
	_(slor, 	"||") \
	_(eq, 		"==") \
	_(neq, 		"!=") \
	_(lt, 		"<") \
	_(le, 		"<=") \
	_(gt, 		">") \
	_(ge, 		">=") \
	_(dollar, 	"$") \
	_(at, 		"@") \
	_(colon, 	":") \
	_(nsget, 	"::") \
	_(nsgetint, 	":::") \
	_(bracket, 	"[") \
	_(bb, 		"[[") \
	_(question, 	"?") \
	_(abs, 		"abs") \
	_(sign, 	"sign") \
	_(sqrt, 	"sqrt") \
	_(floor, 	"floor") \
	_(ceiling, 	"ceiling") \
	_(trunc, 	"trunc") \
	_(round, 	"round") \
	_(signif, 	"signif") \
	_(exp, 		"exp") \
	_(log, 		"log") \
	_(cos, 		"cos") \
	_(sin, 		"sin") \
	_(tan, 		"tan") \
	_(acos, 	"acos") \
	_(asin, 	"asin") \
	_(atan, 	"atan") \
	_(sum,		"sum") \
	_(prod,		"prod") \
	_(min,		"min") \
	_(max,		"max") \
	_(any,		"any") \
	_(all,		"all") \
	_(cumsum,	"cumsum") \
	_(cumprod,	"cumprod") \
	_(cummin,	"cummin") \
	_(cummax,	"cummax") \
	_(cumany,	"cumany") \
	_(cumall,	"cumall") \
	_(names, 	"names") \
	_(dim, 		"dim") \
	_(classSym, 	"class") \
	_(bracketAssign, "[<-") \
	_(bbAssign, 	"[[<-") \
	_(dollarAssign, "$<-") \
	_(UseMethod, 	"UseMethod") \
	_(seq_len, 	"seq_len") \
	_(True, 	"TRUE") \
	_(False, 	"FALSE") \
	_(type, 	"typeof") \
	_(length, 	"length") \
	_(value, 	"value") \
	_(dotGeneric, 	".Generic") \
	_(dotMethod, 	".Method") \
	_(dotClass, 	".Class") \
	_(docall,	"do.call") \
	_(list,		"list") \
	_(missing,	"missing") \
	_(quote,	"quote") \
	_(mmul,		"%*%") \
	_(apply,	"apply") \

struct String {
	int64_t i;
	bool operator==(String o) const { return i == o.i; }
	bool operator!=(String o) const { return i != o.i; }
	bool operator<(String o) const { return i < o.i; }
	bool operator>(String o) const { return i > o.i; }
	static String Init(char const* i) { return (String){(int64_t)i}; }
};

namespace Strings {
       #define CONST_DECLARE(name, string, ...) static const ::String name = String::Init(string);
       STRINGS(CONST_DECLARE)
       #undef CONST_DECLARE
}

struct StringHash {
	size_t operator()( const String& x ) const {
		return x.i>>3;
	} 
};

class StringTable {
	//std::map<std::string, String> stringTable;
	tbb::concurrent_unordered_map<std::string, String> stringTable;
public:
	StringTable() {
	#define ENUM_STRING_TABLE(name, string) \
		stringTable[string] = Strings::name; \

		STRINGS(ENUM_STRING_TABLE);
	}

	String in(std::string const& s) {
		//std::map<std::string, String>::const_iterator i = stringTable.find(s);
		tbb::concurrent_unordered_map<std::string, String, StringHash>::const_iterator i = stringTable.find(s);
		if(i == stringTable.end()) {
			char* str = new char[s.size()+1];
			memcpy(str, s.c_str(), s.size()+1);
			String string = String::Init(str);
			stringTable[s] = string;
			return string;
		} else return i->second;
	}

	std::string out(String i) const {
		if(i.i < 0) return std::string("..") + intToStr(-i.i);
		else return std::string((char const*)i.i);
	}
};

#endif
