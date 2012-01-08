
#ifndef _RIPOSTE_SYMBOLS_H
#define _RIPOSTE_SYMBOLS_H

#include <map>
#include "common.h"
#include "thread.h"

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

typedef const char* String;

namespace Strings {
	static const String NA = 0;
#define DECLARE(name, string, ...) extern String name;
	STRINGS(DECLARE)
#undef DECLARE
}

// TODO: Make this use a good concurrent map implementation 
class StringTable {
	std::map<std::string, String> stringTable;
	Lock lock;
public:
	StringTable() {
	#define ENUM_STRING_TABLE(name, string) \
		stringTable[string] = Strings::name; 
		STRINGS(ENUM_STRING_TABLE);
	}

	String in(std::string const& s) {
		lock.acquire();
		std::map<std::string, String>::const_iterator i = stringTable.find(s);
		if(i == stringTable.end()) {
			char* str = new char[s.size()+1];
			memcpy(str, s.c_str(), s.size()+1);
			String string = (String)str;
			stringTable[s] = string;
			lock.release();
			return string;
		} else {
			lock.release();
			return i->second;
		}
	}

	std::string out(String s) const {
		if((int64_t)s < 0) return std::string("..") + intToStr(-(int64_t)s);
		else return std::string(s);
	}
};

#endif
