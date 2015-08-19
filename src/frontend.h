
#ifndef RIPOSTE_FRONT_END
#define RIPOSTE_FRONT_END

#include "interpreter.h"

// Some code used by the parser and compiler that
// needs to know about certain attributes (names and class).
// Here to keep the core VM clean of this stuff.

inline bool hasNames(Object const& v) {
	return v.attributes()->get(Strings::names);
}

inline bool hasClass(Object const& v) {
	return v.attributes()->get(Strings::classSym);
}

inline Value const& getNames(Object const& v) {
	return *v.attributes()->get(Strings::names);
}

inline Value const& getClass(Object const& v) {
	return *v.attributes()->get(Strings::classSym);
}

inline String className(Object const& o) {
	if(hasClass(o)) {
		Value const& v = getClass(o);
		if(v.isCharacter() && ((Character const&)v).length() > 0)
			return ((Character const&)v)[0];
	}
	return Strings::NA;
}

inline String baseClassName(Object const& o) {
	if(hasClass(o)) {
		Value const& v = getClass(o);
		if(v.isCharacter() && ((Character const&)v).length() > 0)
			return ((Character const&)v)[((Character const&)v).length()-1];
	}
	return Strings::NA;
}

inline bool isSymbol(Value const& v) {
        return 	v.isObject()
		&& hasClass((Object const&)v) 
		&& baseClassName((Object const&)v) == Strings::name;
}

inline bool isCall(Value const& v) {
        return 	v.isObject()
		&& hasClass((Object const&)v) 
		&& baseClassName((Object const&)v) == Strings::call;
}

inline bool isExpression(Value const& v) {
        return 	v.isObject()
		&& hasClass((Object const&)v) 
		&& baseClassName((Object const&)v) == Strings::expression;
}

inline bool isPairlist(Value const& v) {
        return 	v.isObject()
		&& hasClass((Object const&)v) 
		&& baseClassName((Object const&)v) == Strings::pairlist;
}

inline Object CreateSymbol(Global& g, String s) {
	Object v = Character::c(s);
    v.attributes(g.symbolDict);
	return v;
}

inline List CreateExpression(Global& g, List l) {
	l.attributes(g.exprDict);
	return l;
}

inline List CreateCall(Global& g, List l, Value const& names = Value::Nil()) {
    if(!names.isNil()) {
    	Dictionary* d = new Dictionary(2);
	    d->insert(Strings::classSym) = Character::c(Strings::call);
		d->insert(Strings::names) = names;
	    l.attributes(d);
    }
    else {
        l.attributes(g.callDict);
    }
	return l;
}

inline List CreatePairlist(List l, Value const& names = Value::Nil()) {
	Dictionary* d = new Dictionary(2);
	d->insert(Strings::classSym) = Character::c(Strings::pairlist);
	if(!names.isNil())
		d->insert(Strings::names) = names;
	l.attributes(d);
	return l;
}

inline List CreateNamedList(List l, Value const& names) {
	l.attributes(new Dictionary(Strings::names, names));
	return l;
}

inline Object CreateComplex(double a) {
	Object l = List::c(Double::c(0), Double::c(a));
	Dictionary* d = new Dictionary(2);
	d->insert(Strings::classSym) = Character::c(Strings::Complex);
	d->insert(Strings::names) = Character::c(Strings::Re, Strings::Im);
	l.attributes(d);
	return l;
}

inline String SymbolStr(Value const& v) {
	assert(v.isCharacter() && ((Character const&)v).length() == 1);
	return ((Character const&)v)[0];
}

#endif
