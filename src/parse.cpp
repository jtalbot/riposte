
#include "rinst.h"
#include "value.h"

void parse(State& state, SEXP s, Value& v);

List parseList(State& state, SEXP s) {
	uint64_t length = Rf_length(s);
	List d(length);
	Character n(length);
	bool named = false;
	for(uint64_t i = 0; i < length; i++) { 
		parse(state, CAR(s), d[i]); 
		if(!Rf_isNull(TAG(s)))
			{n[i] = state.inString(std::string(CHAR(PRINTNAME(TAG(s))))); named=true;} 
		else n[i] = 0; /* 0 == "" */  
		s = CDR(s); 
	}
	if(named) {
		Value v;
		n.toValue(v);
		setNames(d.attributes, v);
	}
	return d;
}

void parse(State& state, SEXP s, Value& v)
{
	if(Rf_isNull(s)) {
		v = Null::singleton;
	} else if(Rf_isExpression(s)) {
		uint64_t length = Rf_length(s);
		Expression d(length);
		for(uint64_t i = 0; i < length; i++) parse(state, VECTOR_ELT(s,i), d[i]);
		d.toValue(v);
	} else if(Rf_isLanguage(s)) {
		Call d(parseList(state, s));
		d.toValue(v);
		if(d[0].type == Type::R_symbol && state.outString(Symbol(d[0]).i) == ".Internal" && d[1].type == Type::R_call)
		{
			Call call(d[1]);
			Call internal(2);
			internal[0] = d[0];
			internal[1] = call[0];
			call[0] = internal;
			call.toValue(v);
		}
	} else if(Rf_isSymbol(s)) {
		Symbol symbol(state, std::string(CHAR(PRINTNAME(s))));
		symbol.toValue(v);
	} else if(Rf_isReal(s)) {
		uint64_t length = Rf_length(s);
		Double d(length);
		memcpy(d.data(), REAL(s), length*sizeof(double));
		d.toValue(v);
	} else if(Rf_isLogical(s)) {
		uint64_t length = Rf_length(s);
		Logical d(length);
		for(uint64_t i = 0; i < length; i++)
			d[i] = LOGICAL(s)[i] != 0;
		d.toValue(v);
	} else if(Rf_isInteger(s)) {
		uint64_t length = Rf_length(s);
		Integer d(length);
		for(uint64_t i = 0; i < length; i++)
			d[i] = INTEGER(s)[i];
		d.toValue(v);
	} else if(Rf_isString(s)) {
		uint64_t length = Rf_length(s);
		Character d(length);
		for(uint64_t i = 0; i < length; i++)
			d[i] = state.inString(std::string(CHAR(STRING_ELT(s,i))));
		d.toValue(v);
	} else if(Rf_isPairList(s)) {
		//printf("Parsing pair list\n");
		PairList(parseList(state, s)).toValue(v);
	} else if(Rf_isList(s)) {
		parseList(state, s).toValue(v);
	}
	else {
        printf("unhandled type: ");
		if(Rf_isList(s))
			printf("list");
		if(Rf_isVector(s))
			printf("vector");
		if(Rf_isVectorAtomic(s))
			printf("vectoratomic");
		if(Rf_isPrimitive(s))
			printf("primitive");
		if(Rf_isInteger(s))
			printf("integer");
		if(Rf_isNumeric(s))
			printf("numeric");
		printf("\n");
	}
}
