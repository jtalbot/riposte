
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

/*
TODO: blocked by boehm not seeing packed pointer.
1 / 12 / 51
5 bits for type
46 bits for pointer...assume 4 byte aligned
*/

class Value {
public:
	union {
		void* p;
		double d;
		int64_t i;
	};
	Type t;
	uint64_t packed:2;
	// promote length to here...
	// object flag?
	// named flag?

public:
	Value() {/*t=Type::NIL;*/}
	Value(Type const& type, void* ptr, uint64_t packed=0) : p(ptr), t(type), packed(packed) {}
	Value(Type const& type, double d, uint64_t packed=1) : d(d), t(type), packed(packed) {}
	Type type() const { return t; }
	void* ptr() const { return p; }
	std::string toString() const;

	// type constructor
	static void set(Value& v, Type const& type, void* ptr, uint64_t packed=2) {
		v.t = type;
		v.p = ptr;
		v.packed = 2;
	}

	// not right for scalar doubles and maybe other stuff, careful!
	bool operator==(Value const& other) {
		return t == other.t && i == other.i;
	}

	static const Value null;
	static const Value NIL;
};

class Symbol {
	static std::map<std::string, uint64_t> symbolTable;
	static std::map<uint64_t, std::string> reverseSymbolTable;

public:

	uint64_t i;

	Symbol() {
		// really need to initialize to the empty symbol here...
		i = 0;
	}

	Symbol(uint64_t index) {
		i = index;
	}

	Symbol(std::string const& name) { 
		if(symbolTable.find(name) == symbolTable.end()) {
			symbolTable[name] = symbolTable.size();
			reverseSymbolTable[symbolTable[name]] = name;
		}
		i = symbolTable[name];
	}

	Symbol(Value const& v) {
		assert(v.type() == Type::R_symbol); 
		i = v.i;
	}

	void toValue(Value& v) {
		v.t = Type::R_symbol;
		v.i = i;
	}

	uint64_t index() const { return i; }

	std::string const& toString() const { return reverseSymbolTable[i]; }
	bool operator<(Symbol const& other) const { return i < other.i;	}
	bool operator==(Symbol const& other) const { return i == other.i; }
	bool operator!=(Symbol const& other) const { return i != other.i; }
};

struct Stack : public gc {
	Value s[1024];
	uint64_t top;
	Stack() : top(0) {}

	void push(Value const& v) {
		assert(top < 1024);
		s[top++] = v;
	}

	Value& peek() {
		return s[top-1];
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

struct Instruction {
	ByteCode bc;
	void const* ibc;
	uint64_t a, b, c;

	Instruction(ByteCode bc, uint64_t a=0, uint64_t b=0, uint64_t c=0) :
		bc(bc), a(a), b(b), c(c) {}
	
	Instruction(void const* ibc, uint64_t a=0, uint64_t b=0, uint64_t c=0) :
		ibc(ibc), a(a), b(b), c(c) {}
	
	std::string toString() const {
		return std::string("") + bc.toString() + "\t" + intToStr(a);
	}
};

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
	Block() { 
		inner = new Inner();
	}
	
	Block(Value const& v) {
		assert(v.type() == Type::I_bytecode || v.type() == Type::I_promise || v.type() == Type::I_sympromise);
		inner = (Inner*)v.p;
	}

	void toValue(Value& v) {
		v.t = Type::I_bytecode;
		v.p = inner;
	}

	std::string toString() const {
		std::string r = "block:\nconstants: " + intToStr(inner->constants.size()) + "\n";
		for(uint64_t i = 0; i < inner->constants.size(); i++)
			r = r + intToStr(i) + "=\t" + inner->constants[i].toString() + "\n";
		
		r = r + "code: " + intToStr(inner->code.size()) + "\n";
		for(uint64_t i = 0; i < inner->code.size(); i++)
			r = r + intToStr(i) + ":\t" + inner->code[i].toString() + "\n";
		
		return r;
	}

	Value const& expression() const { return inner->expression; }
	Value& expression() { return inner->expression; }
	std::vector<Value, traceable_allocator<Value> > const& constants() const { return inner->constants; }
	std::vector<Value, traceable_allocator<Value> >& constants() { return inner->constants; }
	std::vector<Instruction> const& code() const { return inner->code; }
	std::vector<Instruction>& code() { return inner->code; }
	std::vector<Instruction>& threadedCode() const { return inner->threadedCode; }
};

class Character;

struct VectorInner : public gc {
	uint64_t width, length;
	Value names;
	void* data;
};

struct Vector {
	Type t;
	union {
		VectorInner* inner;
		int64_t pack;
	};
	uint64_t packed;	
	
	uint64_t length() const { if(packed < 2) return packed; else return inner->length; }
	void* data() const { if(packed < 2 ) return (void*)&pack; else return inner->data; }

	Vector() {
	}

	Vector(Value const& v) {
		t = v.type();
		pack = v.i;
		packed = v.packed;
	}

	Vector(Type const& t, uint64_t length, bool named=false);

	void toValue(Value& v) const {
		v.t = t;
		v.i = pack;
		v.packed = packed;
	}
};

template<Type::Value VectorType, class ElementType>
struct VectorImpl {
	VectorInner* inner;
	
	ElementType& operator[](uint64_t index) { return ((ElementType*)(inner->data))[index]; }
	ElementType const& operator[](uint64_t index) const { return ((ElementType*)(inner->data))[index]; }
	uint64_t length() const { return inner->length; }
	ElementType* data() const { return (ElementType*)inner->data; }

	VectorImpl(uint64_t length, bool named=false) {
		inner = new VectorInner();
		inner->length = length;
		inner->width = sizeof(ElementType);
		inner->data = new (GC) ElementType[length];
	}

	VectorImpl(Value const& v) {
		assert(v.type() == VectorType); 
		inner = (VectorInner*)v.p;
	}
	
	VectorImpl(Vector const& v) {
		assert(v.t == VectorType); 
		inner = (VectorInner*)v.inner;
	}

	void toValue(Value& v) const {
		v.t = VectorType;
		v.p = inner;
		v.packed = 2;
	}

	void toVector(Vector& v) const {
		v.t = VectorType;
		v.inner = inner;
		v.packed = 2;
	}

	Value const& names() const {
		return inner->names;
	}

protected:
	VectorImpl() {inner = 0;}
};

template<Type::Value VectorType, class ElementType>
struct PackedVectorImpl {
	union {
		VectorInner* inner;
		int64_t pack;
	};
	uint64_t packed;
	
	ElementType& operator[](uint64_t index) { if(packed < 2) return ((ElementType*)&pack)[index]; else return ((ElementType*)(inner->data))[index]; }
	ElementType const& operator[](uint64_t index) const { if(packed < 2) return ((ElementType*)&pack)[index]; else return ((ElementType*)(inner->data))[index]; }
	uint64_t length() const { if(packed < 2) return packed; else return inner->length; }
	ElementType* data() const { if(packed < 2 ) return (ElementType*)&pack; else return (ElementType*)inner->data; }

	PackedVectorImpl(uint64_t length, bool named=false) {
		if(length >=2 || named) {
			inner = new VectorInner();
			inner->length = length;
			inner->width = sizeof(ElementType);
			inner->data = new (GC) ElementType[length];
			packed = 2;
		} else {
			packed = length;
		}
	}

	PackedVectorImpl(Value const& v) {
		assert(v.type() == VectorType); 
		pack = v.i;
		packed = v.packed;
	}
	
	PackedVectorImpl(Vector const& v) {
		assert(v.t == VectorType); 
		pack = v.pack;
		packed = v.packed;
	}

	void toValue(Value& v) const {
		v.t = VectorType;
		v.i = pack;
		v.packed = packed;
	}

	void toVector(Vector& v) const {
		v.t = VectorType;
		v.pack = pack;
		v.packed = packed;
	}

	Value const& names() const {
		if(packed >= 2)
			return inner->names;
		else
			return Value::null;
	}

	/*void subset(uint64_t start, uint64_t length, Value& v) const {
		VectorInner* i = new VectorInner();
		i->length = length;
		i->width = inner->width;
		i->data = new (GC) ElementType[length];
		for(uint64_t j = start; j < start+length; j++)
			((ElementType*)(i->data))[j-start] = (*this)[j];
		v.p = i;
		v.t = VectorType;
	}*/

protected:
	PackedVectorImpl() {inner = 0;packed=2;}
};


struct complex {
	double a,b;
};

#define VECTOR_IMPL(Super, Name, Type, type) \
struct Name : public Super<Type, type> \
	{ Name(uint64_t length, bool named=false) : Super<Type, type>(length, named) {} \
	  Name(Value const& v) : Super<Type, type>(v) {} \
	  Name(Vector const& v) : Super<Type, type>(v) {}
/* note missing }; */

VECTOR_IMPL(PackedVectorImpl, Logical, Type::R_logical, unsigned char) };
VECTOR_IMPL(PackedVectorImpl, Integer, Type::R_integer, int64_t) };
VECTOR_IMPL(PackedVectorImpl, Double, Type::R_double, double) };
VECTOR_IMPL(VectorImpl, Complex, Type::R_complex, complex) };
VECTOR_IMPL(VectorImpl, Character, Type::R_character, std::string) };
VECTOR_IMPL(PackedVectorImpl, Raw, Type::R_raw, unsigned char) };
VECTOR_IMPL(VectorImpl, List, Type::R_list, Value) };
VECTOR_IMPL(VectorImpl, Call, Type::R_call, Value) 
	Call(List const& v) { inner = v.inner; } };
VECTOR_IMPL(VectorImpl, InternalCall, Type::I_internalcall, Value) 
	InternalCall(List const& v) { inner = v.inner; }  
	InternalCall(Call const& v) { inner = v.inner; } }; 
VECTOR_IMPL(VectorImpl, Expression, Type::R_expression, Value) 
	Expression(List const& v) { inner = v.inner; }  };

inline Vector::Vector(Type const& t, uint64_t length, bool named) {
	switch(t.internal()) {
		case Type::R_logical: Logical(length, named).toVector(*this); break;	
		case Type::R_integer: Integer(length, named).toVector(*this); break;	
		case Type::R_double: Double(length, named).toVector(*this); break;	
		case Type::R_complex: Complex(length, named).toVector(*this); break;	
		case Type::R_character: Character(length, named).toVector(*this); break;	
		case Type::R_raw: Raw(length, named).toVector(*this); break;	
		case Type::R_list: List(length, named).toVector(*this); break;	
		case Type::R_call: Call(length, named).toVector(*this); break;	
		case Type::I_internalcall: InternalCall(length, named).toVector(*this); break;	
		case Type::R_expression: Expression(length, named).toVector(*this); break;	
		default: printf("Invalid vector type\n"); assert(false); break;
	};
}

class Environment;

class Function {
private:
	struct Inner : public gc {
		List args;
		Value body;		// Not necessarily a Block consider function(x) 2, body is the constant 2
		Environment* s;
		Inner(List const& args, Value const& body, Environment* s) : args(args), body(body), s(s) {}
	};
	
	Inner* inner;

public:
	Function(List const& args, Value const& body, Environment* s) {
		inner = new Inner(args, body, s);
	}
	Function(Value const& v) {
		assert(v.t == Type::R_function);
		inner = (Inner*)v.ptr(); 
	}
	void toValue(Value& v) const {
		v.t = Type::R_function;
		v.p = inner;
	}

	List const& args() const { return inner->args; }
	Value const& body() const { return inner->body; }
	Environment* s() const { return inner->s; }
};

// The riposte state
struct State {
	Stack* stack;
	Environment *env, *baseenv;

	// ... string tables

	State(Stack* stack, Environment* env, Environment* baseenv) {
		this->stack = stack;
		this->env = env;
		this->baseenv = baseenv;
	}
};


class CFunction {
public:
typedef uint64_t (*Cffi)(State& s, uint64_t nargs);
	Cffi func;
	CFunction(Cffi func) : func(func) {}
	CFunction(Value const& v) {
		assert(v.t == Type::R_cfunction);
		func = (Cffi)v.ptr(); 
	}
	void toValue(Value& v) const {
		v.t = Type::R_cfunction;
		v.p = (void*)func;
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
	//bool expanded;
	mutable Container container;
	//struct entry { Symbol s; Value v; };
	//mutable entry a[32];

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
	/*int64_t indexOf(Symbol const& name) const {
		uint64_t key = name.index() & 0x1F;
		while(a[key].v.type() != Type::I_nil && a[key].s != name) {
			key = (key+1) & 0x1F;
		}
		return key;
	}*/

public:
	Environment() : s(0), d(0), size(0)/*, expanded(false)*/ {}
	Environment(Environment* s, Environment* d) : s(s), d(d), size(0) /*, expanded(false)*/ {
		/*for(uint64_t i = 0; i < 32; i++)
			Value::set(a[i].v, Type::I_nil, 0);
		*/
	}
	
 	void init(Environment* s, Environment* d) {
		this->s = s;
		this->d = d;
		//this->expanded = false;
		this->size = 0;
		/*for(uint64_t i = 0; i < 32; i++)
			Value::set(a[i].v, Type::I_nil, 0); */
	}

	Environment* dynamic() const {
		return this->d;
	}

	bool getRaw(Symbol const& name, Value& value) const {
		/*if(!expanded) {
			int64_t key = indexOf(name);
			if(a[key].v.type() != Type::I_nil) {
				value = a[key].v;
				return true;
			}
		}
		else*/ if(container.find(name) != container.end()) {
			value = container.find(name)->second;
			return true;
		}
		return false;
	}

	void get(State& state, Symbol const& name, Value& value) const {
		/*if(!expanded) {
			int64_t key = indexOf(name);
			if(a[key].v.type() != Type::I_nil) {
				value = a[key].v;
				if(value.type() == Type::I_promise) {
					eval(state, Block(value), d);
					// This is a redundent copy, eliminate
					value = state.stack->pop();
				} else if(value.type() == Type::I_sympromise) {
					d->get(state, Symbol(Block(value).code()[0].a), value);
				}
				a[key].v = value;
				return;
			}
		}
		else*/ if(container.find(name) != container.end()) {
			value = container.find(name)->second;
			if(value.type() == Type::I_promise) {
				eval(state, Block(value), d);
				// This is a redundent copy, eliminate
				value = state.stack->pop();
			} else if(value.type() == Type::I_sympromise) {
				d->get(state, Symbol(Block(value).code()[0].a), value);
			}
			container[name] = value;
			return;
		}
		if(s != 0) { 
			s->get(state, name, value);
		} else {
			std::cout << "Unable to find object " << name.toString() << std::endl; 
			value = Value::null;
		}
	}

	void getQuoted(Symbol const& name, Value& value) const {
		getRaw(name, value);
		if(value.type() == Type::I_promise || value.type() == Type::I_sympromise) {
			value = Block(value).expression();
		}
	}

	void getCode(Symbol const& name, Block& block) const {
		Value value;
		getRaw(name, value);
		block = Block(value);
	}

	Value assign(Symbol const& name, Value const& value) {
		//printf("Assigning %s into %d\n", name.toString().c_str(), this);
		/*if(!expanded) {
			int64_t key = indexOf(name);
			if(a[key].v.type() == Type::I_nil)
				size++;
			a[key].s = name;
			a[key].v = value;
		} else*/ {
			if(container.find(name) == container.end())
				size++;
			container[name] = value;
		}

		/*if(size > 24) {
			expanded = true;
			for(uint64_t i = 0; i < 32; i++)
				if(a[i].v.type() != Type::I_nil)
					container[a[i].s] = a[i].v;
		}*/

		return value;
	}

	void rm(Symbol const& name) {
		/*if(!expanded) {
			int64_t key = indexOf(name);
			if(a[key].v.type() != Type::I_nil)
				size--;
			Value::set(a[key].v, Type::I_nil, 0);
		} else*/ {
			if(container.find(name) != container.end())
				size--;
			container[name] = Value::NIL;
		}
	}
};


inline double asReal1(Value const& v) { assert(v.type() == Type::R_double || v.type() == Type::R_integer); if(v.type() == Type::R_integer) return Integer(v)[0]; else return Double(v)[0]; }

#endif
