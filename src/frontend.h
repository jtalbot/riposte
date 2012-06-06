
#ifndef RIPOSTE_FRONT_END
#define RIPOSTE_FRONT_END

// Some code used by the parser and compiler that
// needs to know about certain attributes (names and class).
// Here to keep the core VM clean of this stuff.

inline bool hasNames(Value const& v) {
	return v.isObject() && ((Object const&)v).has(Strings::names);
}

inline bool hasClass(Value const& v) {
	return v.isObject() && ((Object const&)v).has(Strings::classSym);
}

inline Value const& getNames(Object const& v) {
	return v.get(Strings::names);
}

inline Value const& getClass(Object const& v) {
	return v.get(Strings::classSym);
}

inline String className(Object const& o) {
	if(hasClass(o)) {
		Value const& v = getClass(o);
		if(v.isCharacter() && v.length > 0)
			return ((Character const&)v)[0];
	}
	return Strings::NA;
}

inline bool isSymbol(Value const& v) {
        return hasClass(v) && className((Object const&)v) == Strings::Symbol;
}

inline bool isCall(Value const& v) {
        return hasClass(v) && className((Object const&)v) == Strings::Call;
}

inline bool isExpression(Value const& v) {
        return hasClass(v) && className((Object const&)v) == Strings::Expression;
}

inline bool isPairlist(Value const& v) {
        return hasClass(v) && className((Object const&)v) == Strings::Pairlist;
}

inline Value CreateSymbol(String s) {
	Object o;
	Object::Init(o, Character::c(s));
	o.insertMutable(Strings::classSym, Character::c(Strings::Symbol));
        return o;
}

inline Value CreateExpression(List const& list) {
	Object o;
	Object::Init(o, list);
	o.insertMutable(Strings::classSym, Character::c(Strings::Expression));
        return o;
}

inline Value CreateCall(List const& list, Value const& names = Value::Nil()) {
        Object o;
	Object::Init(o, list);
	o.insertMutable(Strings::classSym, Character::c(Strings::Call));
	o.insertMutable(Strings::names, names);
        return o;
}

inline Value CreatePairlist(List const& list, Value const& names = Value::Nil()) {
        Object o;
	Object::Init(o, list);
	o.insertMutable(Strings::classSym, Character::c(Strings::Pairlist));
	o.insertMutable(Strings::names, names);
        return o;
}

inline String SymbolStr(Value const& v) {
        if(v.isObject()) return Character(((Object const&)v).base())[0];
        else return Character(v)[0];
}

inline Value CreateComplex(double d) {
	Object o;
	Object::Init(o, List::c(Double::c(0), Double::c(d)));
	o.insertMutable(Strings::names, Character::c(Strings::Re, Strings::Im));
	o.insertMutable(Strings::classSym, Character::c(Strings::Complex));
	return o;
}

#endif
