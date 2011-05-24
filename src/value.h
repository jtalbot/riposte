
#ifndef _RIPOSTE_VALUE_H
#define _RIPOSTE_VALUE_H

#include <map>
#include <iostream>
#include <vector>
#include <deque>
#include <assert.h>

#include <gc/gc_cpp.h>
#include <gc/gc_allocator.h>

#include "type.h"
#include "bc.h"
#include "common.h"
#include "enum.h"
#include "symbols.h"
#include "exceptions.h"

#define STATE_NUM_REGISTERS 1024

struct Attributes;

/** Basic value type */
struct Value {
	// Careful: for efficiency, Value is not initialized by default.
	
	union {
		void* p;
		double d;
		int64_t i;
	};
	Attributes* attributes;
	Type type;
	uint64_t packed:2;

	// not right for scalar doubles and maybe other stuff, careful!
	bool operator==(Value const& other) {
		return type == other.type && i == other.i;
	}
	
	bool operator!=(Value const& other) {
		return type != other.type || i == other.i;
	}

	static void set(Value& v, Type type, void* p) {
		v.type = type;
		v.p = p;
		v.attributes = 0;
		v.packed = 0;
	}

	static const Value NIL;
};

class Environment;

// The riposte state
struct State {
	Value Registers[STATE_NUM_REGISTERS];
	Value* registers;
	
	Environment *env, *baseenv;

	// reset at the beginning of a call to eval
	std::vector<std::string> warnings;

	std::map<std::string, uint64_t> stringTable;
	std::map<uint64_t, std::string> reverseStringTable;
	
	Value loopVector;
	int64_t loopIndex;
	int64_t loopStep;
	int64_t loopEnd;

	State(Environment* env, Environment* baseenv);

	uint64_t inString(std::string const& s) {
		if(stringTable.find(s) == stringTable.end()) {
			uint64_t index = stringTable.size()+1; // due to NA string which needs an index, but not an actual entry in the table.
			stringTable[s] = index;
			reverseStringTable[index] = s;
			return index;
		} else return stringTable[s];
	
	}

	std::string const& outString(uint64_t i) const {
		return reverseStringTable.find(i)->second;
	}

	std::string stringify(Value const& v) const;
};


// Value type implementations

struct Symbol {
	uint64_t i;

#define DECLARE_SYMBOLS(EnumType,ENUM_DEF) \
  enum EnumValue { \
    ENUM_DEF(ENUM_VALUE,0) \
  }; \
  ENUM_DEF(ENUM_CONST,EnumType) \

	DECLARE_SYMBOLS(Symbol,SYMBOLS_ENUM)

	Symbol() : i(1) {}						// Defaults to the empty symbol...
	Symbol(uint64_t index) : i(index) {}
	Symbol(State& state, std::string const& s) : i(state.inString(s)) {}

	Symbol(Value const& v) {
		assert(v.type == Type::R_symbol); 
		i = v.i;
	}

	void toValue(Value& v) const {
		v.type = Type::R_symbol;
		v.i = i;
	}

	operator Value() const {
		Value v;
		toValue(v);
		return v;
	}

	std::string const& toString(State const& state) const { return state.outString(i); }
	bool operator<(Symbol const& other) const { return i < other.i;	}
	bool operator==(Symbol const& other) const { return i == other.i; }
	bool operator!=(Symbol const& other) const { return i != other.i; }

	bool operator==(uint64_t other) const { return i == other; }

	uint64_t Enum() const { return i; }

	bool isAssignable() const {
		return !(*this == Symbol::NA || *this == Symbol::empty);
	}
};


// A generic vector representation. Only provides support to access length and void* of data.
struct Vector {
	struct Inner : public gc {
		uint64_t length, width;
		void* data;
	};

	union {
		Inner* inner;
		int64_t i;
	};
	Attributes* attributes;
	Type type;
	uint64_t packed:2;
	
	uint64_t length() const { if(packed < 2) return packed; else return inner->length; }
	uint64_t width() const { if(packed < 2) return /* maximum possible width */ 8; else return inner->width; }
	void* data() const { if(packed < 2 ) return (void*)&i; else return inner->data; }

	Vector() : inner(0), attributes(0), type(Type::I_nil), packed(0) {}

	Vector(Value const& v) {
		assert(isVector(v.type));
		i = v.i;
		attributes = v.attributes;
		type = v.type;
		packed = v.packed;
	}

	Vector(Type const& t, uint64_t length);

	operator Value() const {
		Value v;
		v.i = i;
		v.attributes = attributes;
		v.type = type;
		v.packed = packed;
		return v;
	}
};

template<Type::EnumValue VectorType, typename ElementType>
struct VectorImpl {
	typedef ElementType Element;
	
	Vector::Inner* inner;
	Attributes* attributes;

	ElementType& operator[](uint64_t index) { return ((ElementType*)(inner->data))[index]; }
	ElementType const& operator[](uint64_t index) const { return ((ElementType*)(inner->data))[index]; }
	uint64_t length() const { return inner->length; }
	ElementType* data() const { return (ElementType*)inner->data; }

	VectorImpl(uint64_t length) : inner(0), attributes(0) {
		inner = new Vector::Inner();
		inner->length = length;
		inner->width = sizeof(Element);
		inner->data = new (GC) Element[length];
	}

	VectorImpl(Value const& v) : inner((Vector::Inner*)v.p), attributes(v.attributes) {
		assert(v.type.Enum() == VectorType); 
	}
	
	VectorImpl(Vector const& v) : inner(v.inner), attributes(v.attributes) {
		assert(v.type.Enum() == VectorType); 
	}

	operator Value() const {
		Value v;
		v.type.v = VectorType;
		v.p = inner;
		v.attributes = attributes;
		v.packed = 3;
		return v;
	}

	operator Vector() const {
		Vector v;
		v.inner = inner;
		v.attributes = attributes;
		v.type = VectorType;
		v.packed = 3;
		return v;
	}

	Type type() const {
		Type t;
		t.v = VectorType;
		return t;
	}

protected:
	VectorImpl() {inner = 0;}
};

template<Type::EnumValue VectorType, typename ElementType>
struct PackedVectorImpl {
	typedef ElementType Element;
	
	union {
		Vector::Inner* inner;
		int64_t i;
		ElementType packedData[sizeof(int64_t)/sizeof(ElementType)];
	};
	Attributes* attributes;
	uint64_t packed:2;

	ElementType& operator[](uint64_t index) { if(packed < 2) return packedData[index]; else return ((ElementType*)(inner->data))[index]; }
	ElementType const& operator[](uint64_t index) const { if(packed < 2) return packedData[index]; else return ((ElementType*)(inner->data))[index]; }
	uint64_t length() const { if(packed < 2) return packed; else return inner->length; }
	ElementType* data() const { if(packed < 2) return &packedData[0]; else return (ElementType*)inner->data; }

	PackedVectorImpl(uint64_t length) : inner(0), attributes(0) {
		if(length < 2)
			packed = length;
		else {
			inner = new Vector::Inner();
			inner->length = length;
			inner->width = sizeof(Element);
			inner->data = new (GC) Element[length];
			packed = 2;
		}
	}

	PackedVectorImpl(Value const& v) : i(v.i), attributes(v.attributes), packed(v.packed) {
		assert(v.type.Enum() == VectorType); 
	}
	
	PackedVectorImpl(Vector const& v) : i(v.i), attributes(v.attributes), packed(v.packed) {
		assert(v.type.Enum() == VectorType); 
	}

	operator Value() const {
		Value v;
		v.type.v = VectorType;
		v.i = i;
		v.attributes = attributes;
		v.packed = packed;
		return v;
	}

	operator Vector() const {
		Vector v;
		v.i = i;
		v.attributes = attributes;
		v.type = VectorType;
		v.packed = packed;
		return v;
	}

	Type type() const {
		Type t;
		t.v = VectorType;
		return t;
	}

protected:
	PackedVectorImpl() {inner = 0;packed=0;}
};

struct _complex {
	double a,b;
	_complex() {}
	_complex(double r, double i) : a(r), b(i) {}
};


union _doublena {
	uint64_t i;
	double d;
};

#define VECTOR_IMPL(Parent, Name, Type, type) \
struct Name : public Parent<Type, type> \
	{ Name(uint64_t length) : Parent<Type, type>(length) {} \
	  Name(Value const& v) : Parent<Type, type>(v) {} \
	  Name(Vector const& v) : Parent<Type, type>(v) {}
/* note missing }; */

VECTOR_IMPL(PackedVectorImpl, Null, Type::E_R_null, unsigned char) 
	const static Null singleton; };
VECTOR_IMPL(PackedVectorImpl, Logical, Type::E_R_logical, unsigned char)
	static Logical c(unsigned char c) { Logical r(1); r[0] = c; return r; }
	static bool isNA(unsigned char c) { return c == NAelement; }
	const static bool CheckNA;
	const static unsigned char NAelement;
	const static Logical NA;
	const static Logical True;
	const static Logical False; };
VECTOR_IMPL(PackedVectorImpl, Integer, Type::E_R_integer, int64_t)
	static Integer c(double c) { Integer r(1); r[0] = c; return r; }
	static bool isNA(int64_t c) { return c == NAelement; }
	const static bool CheckNA;
	const static int64_t NAelement;
	const static Integer NA; };
VECTOR_IMPL(PackedVectorImpl, Double, Type::E_R_double, double)
	static Double c(double c) { Double r(1); r[0] = c; return r; }
	static bool isNA(double c) { _doublena a, b; a.d = c; b.d = NAelement; return a.i==b.i; }
	const static bool CheckNA;
	const static double NAelement;
	const static Double NA;
	const static Double Inf;
	const static Double NaN; };
VECTOR_IMPL(VectorImpl, Complex, Type::E_R_complex, _complex)
	static Complex c(double r, double i=0) { Complex c(1); c[0] = _complex(r,i); return c; }
	static Complex c(_complex c) { Complex r(1); r[0] = c; return r; } };
VECTOR_IMPL(VectorImpl, Character, Type::E_R_character, Symbol)
	static Character c(State& state, std::string const& s) { Character c(1); c[0] = Symbol(state, s); return c; }
	static Character c(Symbol const& s) { Character c(1); c[0] = s; return c; }
	static bool isNA(Symbol const& c) { return c == Symbol::NA; }
	const static bool CheckNA;
	const static Symbol NAelement;
	const static Character NA; };
VECTOR_IMPL(PackedVectorImpl, Raw, Type::E_R_raw, unsigned char) };
VECTOR_IMPL(VectorImpl, List, Type::E_R_list, Value) 
	static List c(Value v0) { List c(1); c[0] = v0; return c; }
	static List c(Value v0, Value v1) { List c(2); c[0] = v0; c[1] = v1; return c; }
	static List c(Value v0, Value v1, Value v2) { List c(3); c[0] = v0; c[1] = v1; c[2] = v2; return c; } };
VECTOR_IMPL(VectorImpl, PairList, Type::E_R_pairlist, Value) 
	PairList(List const& v) { inner = v.inner; attributes = v.attributes; }
	static PairList c(Value v0) { PairList c(1); c[0] = v0; return c; }
	static PairList c(Value v0, Value v1) { PairList c(2); c[0] = v0; c[1] = v1; return c; }
	static PairList c(Value v0, Value v1, Value v2) { PairList c(3); c[0] = v0; c[1] = v1; c[2] = v2; return c; }
	operator List() {Value v = *this; v.type = Type::R_list; return List(v);} };
VECTOR_IMPL(VectorImpl, Call, Type::E_R_call, Value) 
	Call(List const& v) { inner = v.inner; attributes = v.attributes; }
	static Call c(Value v0) { Call c(1); c[0] = v0; return c; }
	static Call c(Value v0, Value v1) { Call c(2); c[0] = v0; c[1] = v1; return c; }
	static Call c(Value v0, Value v1, Value v2) { Call c(3); c[0] = v0; c[1] = v1; c[2] = v2; return c; }
	static Call c(Value v0, Value v1, Value v2, Value v3) { Call c(4); c[0] = v0; c[1] = v1; c[2] = v2; c[3] = v3; return c; } };
VECTOR_IMPL(VectorImpl, Expression, Type::E_R_expression, Value) 
	Expression(List const& v) { inner = v.inner; attributes = v.attributes; }
	static Expression c(Value v0) { Expression c(1); c[0] = v0; return c; }
	static Expression c(Value v0, Value v1) { Expression c(2); c[0] = v0; c[1] = v1; return c; }
	static Expression c(Value v0, Value v1, Value v2) { Expression c(3); c[0] = v0; c[1] = v1; c[2] = v2; return c; } };

inline Vector::Vector(Type const& t, uint64_t length) {
	switch(t.Enum()) {
		case Type::E_R_logical: *this = Logical(length); break;	
		case Type::E_R_integer: *this = Integer(length); break;	
		case Type::E_R_double: *this = Double(length); break;	
		case Type::E_R_complex: *this = Complex(length); break;	
		case Type::E_R_character: *this = Character(length); break;	
		case Type::E_R_raw: *this = Raw(length); break;	
		case Type::E_R_list: *this = List(length); break;	
		case Type::E_R_pairlist: *this = PairList(length); break;	
		case Type::E_R_call: *this = Call(length); break;	
		case Type::E_R_expression: *this = Expression(length); break;	
		default: throw RuntimeError("attempt to create invalid vector type"); break;
	};
}

class Closure {
private:
	struct Inner : public gc {
		Value expression;
		std::vector<Value, traceable_allocator<Value> > constants;
		std::vector<Instruction> code;
		mutable std::vector<Instruction> threadedCode;
	};
	
	Inner* inner;
	Environment* env;	// if NULL, execute in current environment

	Closure(Closure& c, Environment* environment) {
		inner = c.inner;
		env = environment;
	} 
public:
	Closure() : inner(new Inner()), env(0) {}
	
	Closure(Value const& v) {
		assert(	v.type == Type::I_closure || 
			v.type == Type::I_promise ||
			v.type == Type::I_default); 
		inner = (Inner*)v.p;
		env = (Environment*)v.attributes;
	}

	void toValue(Value& v) const {
		v.type = Type::I_closure;
		v.p = inner;
		v.attributes = (Attributes*)env;
	}

	operator Value() const {
		Value v;
		toValue(v);
		return v;
	}

	Closure bind(Environment* environment) {
		return Closure(*this, environment);
	}

	Value const& expression() const { return inner->expression; }
	Value& expression() { return inner->expression; }
	std::vector<Value, traceable_allocator<Value> > const& constants() const { return inner->constants; }
	std::vector<Value, traceable_allocator<Value> >& constants() { return inner->constants; }
	std::vector<Instruction> const& code() const { return inner->code; }
	std::vector<Instruction>& code() { return inner->code; }
	std::vector<Instruction>& threadedCode() const { return inner->threadedCode; }
	Environment* environment() const { return env; }
};


class Function {
private:
	struct Inner : public gc {
		List parameters;
		uint64_t dots;
		Value body;		// Not necessarily a Closure consider function(x) 2, body is the constant 2
		Character str;
		Environment* s;
		Inner(List const& parameters, Value const& body, Character const& str, Environment* s) 
			: parameters(parameters), body(body), str(str), s(s) {}
	};
	
	Inner* inner;
	
public:
	Attributes* attributes;

	Function(List const& parameters, Value const& body, Character const& str, Environment* s); 
	
	Function(Value const& v) {
		assert(v.type == Type::R_function);
		inner = (Inner*)v.p; 
		attributes = v.attributes;
	}

	void toValue(Value& v) const {
		v.p = inner;
		v.attributes = attributes;
		v.type = Type::R_function;
	}

	operator Value() const {
		Value v;
		toValue(v);
		return v;
	}

	List const& parameters() const { return inner->parameters; }
	uint64_t dots() const { return inner->dots; }
	Value const& body() const { return inner->body; }
	Character const& str() const { return inner->str; }
	Environment* s() const { return inner->s; }
};

class CFunction {
public:
	typedef uint64_t (*Cffi)(State& s, Call const& call, List const& args);
	Cffi func;
	CFunction(Cffi func) : func(func) {}
	CFunction(Value const& v) {
		assert(v.type == Type::R_cfunction);
		func = (Cffi)v.p; 
	}
	void toValue(Value& v) const {
		v.type = Type::R_cfunction;
		v.p = (void*)func;
		v.attributes = 0;
	}
};

void eval(State& state, Closure const& closure);

class CompiledCall {
	struct Inner : public gc {
		Value call;
		Value parameters; // a list of closures
		uint64_t dots;
	};

	Inner* inner;
public:
	CompiledCall(Call const& call, State& state);

	CompiledCall(Value const& v) {
		assert(v.type == Type::I_compiledcall);
		inner = (Inner*)v.p; 
	}

	void toValue(Value& v) const {
		v.p = inner;
		v.type = Type::I_compiledcall;
	}

	operator Value() const {
		Value v;
		toValue(v);
		return v;
	}
	
	Call call() const { return Call(inner->call); }
	List parameters() const { return List(inner->parameters); }
	uint64_t dots() const { return inner->dots; }
};

class Environment : public gc {
private:
	Environment *s, *d;		// static and dynamic scopes respectively
	typedef std::map<Symbol, Value, std::less<Symbol>, gc_allocator<std::pair<Symbol, Value> > > Container;
	Container container;
public:
	Environment() : s(0), d(0) {}
	Environment(Environment* s, Environment* d) : s(s), d(d) {}

	Environment* staticParent() const { return s; }
	Environment* dynamicParent() const { return d; }
	
 	void init(Environment* s, Environment* d) {
		this->s = s;
		this->d = d;
	}

	bool getRaw(Symbol const& name, Value& value) const {
		Container::const_iterator i = container.find(name);
		if(i != container.end()) {
			value = i->second;
			return true;
		}
		return false;
	}

	bool get(State& state, Symbol const& name, Value& value) {
		Container::const_iterator i = container.find(name);
		if(i != container.end()) {
			value = i->second;
			if(value.type == Type::I_promise) {
				while(value.type == Type::I_promise) {
					eval(state, Closure(value));
					value = state.registers[0];
				}
				container[name] = value;
			} else if(value.type == Type::I_default) {
				eval(state, Closure(value).bind(this));
				container[name] = value = state.registers[0];
			}
			return true;
		} else if(s != 0) {
			return s->get(state, name, value);
		} else {
			value = Null::singleton;
			return false;
		}
	}

	void getQuoted(Symbol const& name, Value& value) const {
		getRaw(name, value);
		if(value.type == Type::I_promise) {
			value = Closure(value).expression();
		}
	}

	void getCode(Symbol const& name, Closure& closure) const {
		Value value;
		getRaw(name, value);
		closure = Closure(value);
	}

	void assign(Symbol const& name, Value const& value) {
		container[name] = value;
	}

	void rm(Symbol const& name) {
		container.erase(name);
	}
};

class REnvironment {
private:
	Environment* env;
public:
	Attributes* attributes;
	REnvironment(Environment* env) : env(env), attributes(0) {
	}
	REnvironment(Value const& v) {
		assert(v.type == Type::R_environment);
		env = (Environment*)v.p;
		attributes = v.attributes;
	}
	void toValue(Value& v) const {
		v.type = Type::R_environment;
		v.p = env;
		v.attributes = attributes;
	}
	operator Value() const {
		Value v;
		toValue(v);
		return v;
	}
	Environment* ptr() const {
		return env;
	}
};

struct Attributes : public gc {
	Value names;
	Value dim;
	Value dimnames;
	Value klass;
	Attributes() : names(Null::singleton), dim(Null::singleton), 
			dimnames(Null::singleton), klass(Null::singleton) {}
};


inline void setNames(Attributes*& attrs, Vector const& names) {
	Attributes* a = new Attributes();
	if(attrs != 0)
		*a = *attrs;
	a->names = names;
	attrs = a;
}

inline Vector getNames(Attributes const* attrs) {
	if(attrs == 0)
		return Null::singleton;
	else return attrs->names;
}

inline void setClass(Attributes*& attrs, Vector const& klass) {
	Attributes* a = new Attributes();
	if(attrs != 0)
		*a = *attrs;
	a->klass = klass;
	attrs = a;
}

inline Vector getClass(Attributes const* attrs) {
	if(attrs == 0)
		return Null::singleton;
	else return attrs->klass;
}

inline void setDim(Attributes*& attrs, Vector const& dim) {
	Attributes* a = new Attributes();
	if(attrs != 0)
		*a = *attrs;
	a->dim = dim;
	attrs = a;
}

inline Vector getDim(Attributes const* attrs) {
	if(attrs == 0)
		return Null::singleton;
	else return attrs->dim;
}

inline bool isObject(Value const& v) {
	return v.attributes != 0 && v.attributes->klass != Null::singleton;
}

struct Pairs {
	struct Pair {
		Symbol n;
		Value v;
	};
	struct Inner : public gc {
		std::deque<Pair, traceable_allocator<Value> > p;
	};
	Inner* inner;
	static Pairs Make() { Pairs p; p.inner = new Inner(); return p; }
	uint64_t length() const { return inner->p.size(); }
	void push_front(Symbol n, Value v) { Pair t = {n, v}; inner->p.push_front(t); }
	void push_back(Symbol n, Value v)  { Pair t = {n, v}; inner->p.push_back(t); }
	const Value& value(uint64_t i) const { return inner->p[i].v; }
	const Symbol& name(uint64_t i) const { return inner->p[i].n; }
	
	operator List() const {
		List l(length());
		for(uint64_t i = 0; i < length(); i++)
			l[i] = value(i);
		bool named = false;
		for(uint64_t i = 0; i < length(); i++) {
			if(name(i).i != 0) {
				named = true;
				break;
			}
		}
		if(named) {
			Character n(length());
			for(uint64_t i = 0; i < length(); i++)
				n[i] = name(i).i;
			setNames(l.attributes, n);
		}
		return l;
	}
};

#endif
