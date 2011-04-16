
#ifndef _RIPOSTE_VALUE_H
#define _RIPOSTE_VALUE_H

#include <map>
#include <iostream>
#include <vector>
#include <assert.h>

#include <gc/gc_cpp.h>
#include <gc/gc_allocator.h>

#include "type.h"
#include "bc.h"
#include "common.h"

struct Attributes;


/** Basic value type */
class Value {
public:
	union {
		void* p;
		double d;
		int64_t i;
	};
	Attributes* attributes;
	Type type;
	uint64_t packed:2;

	Value() {/*t=Type::NIL;*/}	// Careful: for efficiency, Value is not initialized by default.
	
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

private:
	Value(Type type, void* p, Attributes* attributes)
		: p(p), attributes(attributes), type(type), packed(0) {}
};

// The execution stack
struct Stack : public gc {
	Value s[1024];
	uint64_t top;
	Stack() : top(0) {}

	void push(Value const& v) {
		assert(top < 1024);
		s[top++] = v;
	}

	Value& peek(uint64_t down=0) {
		return s[top-1-down];
	}

	Value& reserve() {
		top++;
		return s[top-1];
	}

	Value pop() {
		assert(top > 0);
		return s[--top];
	}
};

class Environment;

#define EMPTY_STRING 0
#define DOTS_STRING 1

// The riposte state
struct State {
	Stack stack;
	Environment *env, *baseenv;

	std::map<std::string, uint64_t> stringTable;
	std::map<uint64_t, std::string> reverseStringTable;

	State(Environment* env, Environment* baseenv) {
		this->env = env;
		this->baseenv = baseenv;

		// insert strings into table that must be at known positions...
		//   the empty string
		stringTable[""] = EMPTY_STRING;
		reverseStringTable[EMPTY_STRING] = "";
		//   the dots string
		stringTable["..."] = DOTS_STRING;
		reverseStringTable[DOTS_STRING] = "...";
	}

	uint64_t inString(std::string const& s) {
		if(stringTable.find(s) == stringTable.end()) {
			uint64_t index = stringTable.size();
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
	Attributes* attributes;

	Symbol() : i(0), attributes(0) {}						// Defaults to the empty symbol...
	Symbol(uint64_t index) : i(index), attributes(0) {}
	Symbol(State& state, std::string const& s) : i(state.inString(s)), attributes(0) {}

	Symbol(Value const& v) {
		assert(v.type == Type::R_symbol); 
		i = v.i;
		attributes = v.attributes;
	}

	void toValue(Value& v) {
		v.type = Type::R_symbol;
		v.i = i;
		v.attributes = attributes;
	}

	operator Value() {
		Value v;
		toValue(v);
		return v;
	}

	std::string const& toString(State& state) { return state.outString(i); }
	bool operator<(Symbol const& other) const { return i < other.i;	}
	bool operator==(Symbol const& other) const { return i == other.i; }
	bool operator!=(Symbol const& other) const { return i != other.i; }
};


// A generic vector representation. Only provides support to access length and void* of data.
struct Vector {
	struct Inner : public gc {
		uint64_t length, width;
		void* data;
	};

	union {
		Inner* inner;
		int64_t packedData;
	};
	Attributes* attributes;
	Type type;
	uint64_t packed:2;
	
	uint64_t length() const { if(packed < 2) return packed; else return inner->length; }
	uint64_t width() const { if(packed < 2) return /* maximum possible width */ 8; else return inner->width; }
	void* data() const { if(packed < 2 ) return (void*)&packedData; else return inner->data; }

	Vector() : inner(0), attributes(0), packed(0) {}

	Vector(Value const& v) {
		assert(isVector(v.type));
		packedData = v.i;
		attributes = v.attributes;
		type = v.type;
		packed = v.packed;
	}

	Vector(Type const& t, uint64_t length);

	void toValue(Value& v) const {
		v.i = packedData;
		v.attributes = attributes;
		v.type = type;
		v.packed = packed;
	}
	
	operator Value() const {
		Value v;
		toValue(v);
		return v;
	}
};

template<Type::Value VectorType, class ElementType, bool Pack>
struct VectorImpl {
	union {
		Vector::Inner* inner;
		int64_t packedData;
	};
	Attributes* attributes;
	uint64_t packed:2;

	typedef ElementType Element;
	
	Element& operator[](uint64_t index) { if(Pack && packed < 2) return ((Element*)&packedData)[index]; else return ((Element*)(inner->data))[index]; }
	Element const& operator[](uint64_t index) const { if(Pack && packed < 2) return ((Element*)&packedData)[index]; else return ((Element*)(inner->data))[index]; }
	uint64_t length() const { if(Pack && packed < 2) return packed; else return inner->length; }
	Element* data() const { if(Pack && packed < 2) return (Element*)&packedData; else return (Element*)inner->data; }

	VectorImpl(uint64_t length) : packedData(0), attributes(0) {
		if(Pack && length < 2)
			packed = length;
		else {
			inner = new Vector::Inner();
			inner->length = length;
			inner->width = sizeof(Element);
			inner->data = new (GC) Element[length];
			packed = 2;
		}
	}

	VectorImpl(Value const& v) : packedData(v.i), attributes(v.attributes), packed(v.packed) {
		assert(v.type == VectorType); 
	}
	
	VectorImpl(Vector const& v) : packedData(v.packedData), attributes(v.attributes), packed(v.packed) {
		assert(v.type == VectorType); 
	}

	void toValue(Value& v) const {
		v.type = VectorType;
		v.i = packedData;
		v.attributes = attributes;
		v.packed = packed;
	}
	
	operator Value() const {
		Value v;
		toValue(v);
		return v;
	}

	void toVector(Vector& v) const {
		v.packedData = packedData;
		v.attributes = attributes;
		v.type = VectorType;
		v.packed = packed;
	}

	operator Vector() const {
		Vector v;
		toVector(v);
		return v;
	}

	Type type() const {
		return VectorType;
	}

	/*void subset(uint64_t start, uint64_t length, Value& v) const {
		VectorInner* i = new VectorInner();
		i->length = length;
		i->width = inner->width;
		i->data = new (GC) Element[length];
		for(uint64_t j = start; j < start+length; j++)
			((Element*)(i->data))[j-start] = (*this)[j];
		v.p = i;
		v.t = VectorType;
	}*/

protected:
	VectorImpl() {inner = 0;packed=0;}
};


struct complex {
	double a,b;
};

#define VECTOR_IMPL(Name, Type, type, Pack) \
struct Name : public VectorImpl<Type, type, Pack> \
	{ Name(uint64_t length) : VectorImpl<Type, type, Pack>(length) {} \
	  Name(Value const& v) : VectorImpl<Type, type, Pack>(v) {} \
	  Name(Vector const& v) : VectorImpl<Type, type, Pack>(v) {}
/* note missing }; */

VECTOR_IMPL(Null, Type::R_null, Value, true) 
	const static Null singleton; };
VECTOR_IMPL(Logical, Type::R_logical, unsigned char, true) };
VECTOR_IMPL(Integer, Type::R_integer, int64_t, true) };
VECTOR_IMPL(Double, Type::R_double, double, true) };
VECTOR_IMPL(Complex, Type::R_complex, complex, false) };
VECTOR_IMPL(Character, Type::R_character, uint64_t, true) };
VECTOR_IMPL(Raw, Type::R_raw, unsigned char, true) };
VECTOR_IMPL(List, Type::R_list, Value, false) };
VECTOR_IMPL(PairList, Type::R_pairlist, Value, false) 
	PairList(List const& v) { inner = v.inner; attributes = v.attributes; packed = v.packed; } };
VECTOR_IMPL(Call, Type::R_call, Value, false) 
	Call(List const& v) { inner = v.inner; attributes = v.attributes; packed = v.packed; } };
VECTOR_IMPL(Expression, Type::R_expression, Value, false) 
	Expression(List const& v) { inner = v.inner; attributes = v.attributes; packed = v.packed; }  };

inline Vector::Vector(Type const& t, uint64_t length) {
	switch(t.internal()) {
		case Type::R_logical: Logical(length).toVector(*this); break;	
		case Type::R_integer: Integer(length).toVector(*this); break;	
		case Type::R_double: Double(length).toVector(*this); break;	
		case Type::R_complex: Complex(length).toVector(*this); break;	
		case Type::R_character: Character(length).toVector(*this); break;	
		case Type::R_raw: Raw(length).toVector(*this); break;	
		case Type::R_list: List(length).toVector(*this); break;	
		case Type::R_pairlist: PairList(length).toVector(*this); break;	
		case Type::R_call: Call(length).toVector(*this); break;	
		case Type::R_expression: Expression(length).toVector(*this); break;	
		default: printf("Invalid vector type\n"); assert(false); break;
	};
}

class Block {
private:
	struct Inner : public gc {
		Value expression;
		std::vector<Value, traceable_allocator<Value> > constants;
		std::vector<Instruction> code;
		mutable std::vector<Instruction> threadedCode;
	};
	
	Inner* inner;
public:
	Block() : inner(new Inner()) {}
	
	Block(Value const& v) {
		assert(	v.type == Type::I_bytecode || 
				v.type == Type::I_promise || 
				v.type == Type::I_sympromise);
		inner = (Inner*)v.p;
	}

	void toValue(Value& v) {
		v.type = Type::I_bytecode;
		v.p = inner;
		v.attributes = 0;
	}

	Value const& expression() const { return inner->expression; }
	Value& expression() { return inner->expression; }
	std::vector<Value, traceable_allocator<Value> > const& constants() const { return inner->constants; }
	std::vector<Value, traceable_allocator<Value> >& constants() { return inner->constants; }
	std::vector<Instruction> const& code() const { return inner->code; }
	std::vector<Instruction>& code() { return inner->code; }
	std::vector<Instruction>& threadedCode() const { return inner->threadedCode; }
};

class Function {
private:
	struct Inner : public gc {
		PairList parameters;
		Value body;		// Not necessarily a Block consider function(x) 2, body is the constant 2
		Character str;
		Environment* s;
		Inner(PairList const& parameters, Value const& body, Character const& str, Environment* s) 
			: parameters(parameters), body(body), str(str), s(s) {}
	};
	
	Inner* inner;
	
public:
	Attributes* attributes;

	Function(PairList const& parameters, Value const& body, Character const& str, Environment* s) 
		: inner(new Inner(parameters, body, str, s)), attributes(0) {}
	
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

	PairList const& parameters() const { return inner->parameters; }
	Value const& body() const { return inner->body; }
	Character const& str() const { return inner->str; }
	Environment* s() const { return inner->s; }
};

class CFunction {
public:
	typedef uint64_t (*Cffi)(State& s, Call const& call);
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


void eval(State& state, Block const& block);
void eval(State& state, Block const& block, Environment* env); 
Block compile(State& state, Value const& expression);

class Environment : public gc {
private:
	Environment *s, *d;		// static and dynamic scopes respectively
	typedef std::map<Symbol, Value, std::less<Symbol>, gc_allocator<std::pair<Symbol, Value> > > Container;
	uint64_t size;
	Container container;
	/*struct Dot {
		Environment* s;
		Symbol name;
		Value value;
	};
	std::vector<Dot> dots;*/
/*
Insights:
-All promises must be evaluated in the dynamic scope, so typically don't need to store env
-If a promise is re-passed, what is actually passed is a new promise to look up the variable name in the other scope.
-Dots are a special case which do need to store env since they can be repassed.
-Consider this example:
f <- function(...) function() sum(...)
f(1,2,3,4,5)()
15
*/
public:
	Environment() : s(0), d(0), size(0) {}
	Environment(Environment* s, Environment* d) : s(s), d(d), size(0) {}
	
 	void init(Environment* s, Environment* d) {
		this->s = s;
		this->d = d;
		this->size = 0;
	}

	Environment* dynamic() const {
		return this->d;
	}

	bool getRaw(Symbol const& name, Value& value) const {
		if(container.find(name) != container.end()) {
			value = container.find(name)->second;
			return true;
		}
		return false;
	}

	bool get(State& state, Symbol const& name, Value& value) {
		if(container.find(name) != container.end()) {
			value = container.find(name)->second;
			if(value.type == Type::I_promise) {
				eval(state, Block(value), d);
				// This is a redundent copy, eliminate
				value = state.stack.pop();
			} else if(value.type == Type::I_sympromise) {
				d->get(state, value.i, value);
			} else if(value.type == Type::I_default) {
				eval(state, Block(value), this);
				value = state.stack.pop();
			} else if(value.type == Type::I_symdefault) {
				get(state, value.i, value);
			}
			container[name] = value;
			return true;
		}
		if(s != 0) { 
			return s->get(state, name, value);
		} else {
			value = Null::singleton;
			return false;
		}
	}

	void getQuoted(Symbol const& name, Value& value) const {
		getRaw(name, value);
		if(value.type == Type::I_promise || value.type == Type::I_sympromise) {
			value = Block(value).expression();
		}
	}

	void getCode(Symbol const& name, Block& block) const {
		Value value;
		getRaw(name, value);
		block = Block(value);
	}

	Value assign(Symbol const& name, Value const& value) {
		if(name.i < 1) {
			printf("Cannot assign to that symbol\n");
		}
		//printf("Assigning %s into %d\n", name.toString().c_str(), this);
		if(container.find(name) == container.end())
			size++;
		container[name] = value;

		return value;
	}

	void rm(Symbol const& name) {
		if(container.find(name) != container.end())
			size--;
		container[name] = Value::NIL;
	}
};

class REnvironment {
private:
	Environment* env;
public:
	Attributes* attributes;
	REnvironment(Value const& v) {
		assert(v.type == Type::R_environment);
		env = (Environment*)v.p;
		attributes = v.attributes;
	}
	void toValue(Value& v) const {
		v.type == Type::R_environment;
		v.p = env;
		v.attributes = attributes;
	}
	Environment* environment() const {
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

inline double asReal1(Value const& v) { assert(v.type == Type::R_double || v.type == Type::R_integer); if(v.type == Type::R_integer) return Integer(v)[0]; else return Double(v)[0]; }

#endif
