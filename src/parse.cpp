
#include "rinst.h"
#include "value.h"

bool isInternalCall(std::string str) {
	return (str[0] == '.' && str[1] >= 'A' && str[1] <= 'Z');
}

void parse(SEXP s, Value& v);

List parseList(SEXP s) {
	uint64_t length = Rf_length(s);
	List d(length, true);
	Character n(length);
	bool named = false;
	for(uint64_t i = 0; i < length; i++) { 
		parse(CAR(s), d[i]); 
		if(!Rf_isNull(TAG(s)))
			{n[i] = CHAR(PRINTNAME(TAG(s))); named=true;} 
		else n[i] = "";  
		s = CDR(s); 
	}
	if(named) n.toValue(d.inner->names);
	return d;
}

void parse(SEXP s, Value& v)
{
	if(Rf_isNull(s)) {
		v = Value::null;
	} else if(Rf_isExpression(s)) {
		uint64_t length = Rf_length(s);
		Expression d(length);
		for(uint64_t i = 0; i < length; i++) parse(VECTOR_ELT(s,i), d[i]);
		d.toValue(v);
	} else if(Rf_isLanguage(s)) {
		Call d(parseList(s));
		d.toValue(v);
		if(d[0].type() == Type::R_symbol && isInternalCall(Symbol(d[0]).toString())) {
			InternalCall ic = InternalCall(d);
			ic.toValue(v);
		}
		// This code means that you can't redefine ".Internal"
		//else if(length == 2 && d[0].type() == Type::R_symbol && Symbol(d[0]).toString() == ".Internal")
		//{
		//	InternalCall ic = InternalCall(Call(d[1]));
		//	ic.toValue(v);
		//}
	} else if(Rf_isSymbol(s)) {
		Symbol symbol(std::string(CHAR(PRINTNAME(s))));
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
			d[i] = std::string(CHAR(STRING_ELT(s,i)));
		d.toValue(v);
	} else if(Rf_isList(s)) {
		List d(parseList(s));
		d.toValue(v);
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
