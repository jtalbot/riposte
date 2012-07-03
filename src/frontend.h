
#ifndef RIPOSTE_FRONT_END
#define RIPOSTE_FRONT_END

// Some code used by the parser and compiler that
// needs to know about certain attributes (names and class).
// Here to keep the core VM clean of this stuff.

inline Dictionary* attrs(Value const& v) {
	assert(v.isConcrete());
	return (Dictionary*)v.z();
}

inline bool hasNames(Value const& v) {
	return v.isObject() && attrs(v)->has(Strings::names);
}

inline bool hasClass(Value const& v) {
	return v.isObject() && attrs(v)->has(Strings::classSym);
}

inline Value const& getNames(Value const& v) {
	return attrs(v)->get(Strings::names);
}

inline Value const& getClass(Value const& v) {
	return attrs(v)->get(Strings::classSym);
}

inline String className(Value const& o) {
	if(hasClass(o)) {
		Value const& v = getClass(o);
		if(v.isCharacter() && ((Character const&)v).length() > 0)
			return ((Character const&)v)[0];
	}
	return Strings::NA;
}

inline bool isSymbol(Value const& v) {
        return hasClass(v) && className(v) == Strings::Symbol;
}

inline bool isCall(Value const& v) {
        return hasClass(v) && className(v) == Strings::Call;
}

inline bool isExpression(Value const& v) {
        return hasClass(v) && className(v) == Strings::Expression;
}

inline bool isPairlist(Value const& v) {
        return hasClass(v) && className(v) == Strings::Pairlist;
}

inline Value CreateSymbol(String s) {
	Value v = Character::c(s);
	Dictionary* d = new Dictionary(1);
	d->insert(Strings::classSym) = Character::c(Strings::Symbol);
	v.z((uint64_t)d);
	return v;
}

inline Value CreateExpression(List l) {
	Dictionary* d = new Dictionary(1);
	d->insert(Strings::classSym) = Character::c(Strings::Expression);
	l.z((uint64_t)d);
	return l;
}

inline Value CreateCall(List l, Value const& names = Value::Nil()) {
	Dictionary* d = new Dictionary(2);
	d->insert(Strings::classSym) = Character::c(Strings::Call);
	if(!names.isNil())
		d->insert(Strings::names) = names;
	l.z((uint64_t)d);
	return l;
}

inline Value CreatePairlist(List l, Value const& names = Value::Nil()) {
	Dictionary* d = new Dictionary(2);
	d->insert(Strings::classSym) = Character::c(Strings::Pairlist);
	if(!names.isNil())
		d->insert(Strings::names) = names;
	l.z((uint64_t)d);
	return l;
}

inline Value CreateComplex(double a) {
	Value l = List::c(Double::c(0), Double::c(a));
	Dictionary* d = new Dictionary(2);
	d->insert(Strings::classSym) = Character::c(Strings::Complex);
	d->insert(Strings::names) = Character::c(Strings::Re, Strings::Im);
	l.z((uint64_t)d);
	return l;
}

inline String SymbolStr(Value const& v) {
	assert(v.isCharacter() && ((Character const&)v).length() == 1);
	return ((Character const&)v)[0];
}

#endif
