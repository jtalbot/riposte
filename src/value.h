
#ifndef _RIPOSTE_VALUE_H
#define _RIPOSTE_VALUE_H

#include <map>
#include <iostream>
#include <vector>
#include <stack>
#include <deque>
#include <assert.h>
#include <limits>
#include <complex>

#include <gc/gc_cpp.h>
#include <gc/gc_allocator.h>

#include "type.h"
#include "bc.h"
#include "common.h"
#include "enum.h"
#include "symbols.h"
#include "exceptions.h"
#include "trace.h"

struct Attributes;

struct Value {
	Type::Enum type:8;
	int64_t length:56;
	union {
		int64_t i;
		void* p;
	};
	union {
		void* env;
		Attributes* attributes;
	};

	static Value Make(Type::Enum type, int64_t length, int64_t data, void* extra) {
		Value v = {type, length, {data}, {extra}};
		return v;
	}

	static Value Make(Type::Enum type, int64_t length, void* data, void* extra) {
		Value v = {type, length, {(int64_t)data}, {extra}};
		return v;
	}

	bool isNil() const { return type == Type::Nil; }
	bool isNull() const { return type == Type::Null; }
	bool isLogical() const { return type == Type::Logical; }
	bool isInteger() const { return type == Type::Integer; }
	bool isDouble() const { return type == Type::Double; }
	bool isComplex() const { return type == Type::Complex; }
	bool isCharacter() const { return type == Type::Character; }
	bool isList() const { return type == Type::List; }
	bool isPairList() const { return type == Type::PairList; }
	bool isCall() const { return type == Type::Call; }
	bool isExpression() const { return type == Type::Expression; }
	bool isSymbol() const { return type == Type::Symbol; }
	bool isPromise() const { return type == Type::Promise; }
	bool isFunction() const { return type == Type::Function; }
	bool isBuiltIn() const { return type == Type::BuiltIn; }
	bool isMathCoerce() const { return isDouble() || isInteger() || isLogical() || isComplex(); }
	bool isLogicalCoerce() const { return isDouble() || isInteger() || isLogical() || isComplex(); }
	bool isVector() const { return isNull() || isLogical() || isInteger() || isDouble() || isComplex() || isCharacter() || isList(); }
	bool isClosureSafe() const { return isNull() || isLogical() || isInteger() || isDouble() || isComplex() || isCharacter() || isSymbol() || (isList() && length==0); }
	bool isListLike() const { return isList() || isPairList() || isCall() || isExpression(); }
	bool isConcrete() const { return type < Type::Promise; }
};

class Environment;
class State;

struct NilValue {
	operator Value() const {
		return Value::Make(Type::Nil, 0, (int64_t)0, 0);
	}
};

static const NilValue Nil = {};

// Value type implementations

struct Symbol {
	int64_t i;

	// make constant members matching built in strings
	#define CONST_DECLARE(name, string, ...) static const Symbol name;
	STRINGS(CONST_DECLARE)
	#undef CONST_DECLARE

	Symbol() : i(1) {} // Defaults to the empty symbol...
	explicit Symbol(int64_t index) : i(index) {}

	explicit Symbol(Value const& v) {
		assert(v.type == Type::Symbol); 
		i = v.i;
	}

	operator Value() const {
		return Value::Make(Type::Symbol, 0, i, 0);
	}

	bool operator<(Symbol const& other) const { return i < other.i;	}
	bool operator==(Symbol const& other) const { return i == other.i; }
	bool operator!=(Symbol const& other) const { return i != other.i; }

	bool operator==(int64_t other) const { return i == other; }

	bool isAssignable() const {
		return !(*this == Symbol::NA || *this == Symbol::empty);
	}
};


// A generic vector representation. Only provides support to access length and void* of data.
struct Vector {
	void* _data;
	int64_t length, width;
	bool _packed;
	Attributes* attributes;
	Type::Enum type;

	bool packed() const { return _packed; }
	void* data() const { if(packed()) return (void*)&_data; else return _data; }
	void* data(int64_t i) const { if(packed()) return ((char*)&_data) + i*width; else return (char*)_data + i*width; }

	Vector() : _data(0), length(0), width(0), _packed(true), attributes(0), type(Type::Nil) {}

	explicit Vector(Value v);
	explicit Vector(Type::Enum t, int64_t length);
	explicit Vector(Type::Enum t, int64_t length, void * data);

	operator Value() const {
		return Value::Make(type, length, _data, attributes);
	}
};

template<Type::Enum VectorType, typename ElementType, bool Recursive>
struct VectorImpl {
	typedef ElementType Element;
	static const int64_t width = sizeof(ElementType); 
	static const Type::Enum type = VectorType;
	static const int64_t packLength = sizeof(ElementType)/sizeof(void*);

	int64_t length;
	ElementType* _data;
	Attributes* attributes;
	ElementType pack[packLength];

	ElementType& operator[](int64_t index) { return _data[index]; }
	ElementType const& operator[](int64_t index) const { return _data[index]; }
	ElementType const* data() const { return _data; }
	ElementType const* data(int64_t index) const { return _data + index; }
	ElementType* data() { return _data; }
	ElementType* data(int64_t index) { return _data + index; }

	// need to override copy constructor and operator= so the data pointer gets updated.
	VectorImpl(VectorImpl<VectorType, ElementType, Recursive> const& other) {
		length = other.length;
		_data = other._data;
		attributes = other.attributes;
		if(packed()) {
                        memcpy(pack, &other.pack, packLength*sizeof(ElementType));
			_data = pack;
		}
	}

	VectorImpl& operator=(VectorImpl const& other) {
		length = other.length;
		_data = other._data;
		attributes = other.attributes;
		if(packed()) {
                        memcpy(pack, &other.pack, packLength*sizeof(ElementType));
			_data = pack;
		}
		return *this;
	}

	// Cross type cast (between vectors with the same storage type), use carefully
	// Mainly used for casting between Lists, Calls, and Expressions
	template<Type::Enum T>
	VectorImpl(VectorImpl<T, ElementType, Recursive> const& other) {
		length = other.length;
		_data = other._data;
		attributes = other.attributes;
		if(packed()) {
                        memcpy(pack, other.pack, packLength*sizeof(ElementType));
			_data = pack;
		}
	}

	explicit VectorImpl(int64_t length=0) 
		: length(length), 
		_data(
			packed() ? pack :
			length == 0 ? 0 : 
			Recursive ? new (GC) Element[length] : new (PointerFreeGC) Element[length]), 
		attributes(0) {}

	explicit VectorImpl(Value v) : length(v.length), _data((ElementType*)v.p), attributes(v.attributes) {
		assert(v.type == VectorType); 
                if(packed()) {
                        memcpy(pack, &v.p, packLength*sizeof(ElementType));
                        _data = pack;
                }
	}

	explicit VectorImpl(Vector v) 
		: length(v.length), _data((ElementType*)v._data), attributes(v.attributes) {
		assert(v.type == VectorType); 
                if(packed()) {
                        memcpy(pack, &v._data, packLength*sizeof(ElementType));
                        _data = pack;
                }
	}

	operator Value() const {
		Value v = Value::Make(VectorType, length, _data, attributes);
                if(packed()) {
                        memcpy(&v.p, pack, sizeof(void*));
                }
		return v;
	}

	operator Vector() const {
		Vector v;
		v._data = _data;
                if(packed()) {
                        memcpy(&v._data, &pack, sizeof(void*));
                }
		v.length = length;
		v.width = width;
		v._packed = packed();
		v.attributes = attributes;
		v.type = VectorType;
		return v;
	}

protected:
	VectorImpl(ElementType* _data, int64_t length, Attributes* attributes) 
		: length(length), _data(_data), attributes(attributes) {}

	bool packed() const {
		return sizeof(ElementType) <= sizeof(void*) && length <= (int64_t)(sizeof(ElementType)/sizeof(void*));
	}

};

union _doublena {
	int64_t i;
	double d;
};

#define VECTOR_IMPL(Name, Element, Recursive) 				\
struct Name : public VectorImpl<Type::Name, Element, Recursive> { 			\
	explicit Name(int64_t length=0) : VectorImpl<Type::Name, Element, Recursive>(length) {} 	\
	explicit Name(Value const& v) : VectorImpl<Type::Name, Element, Recursive>(v) {} 	\
	explicit Name(Vector const& v) : VectorImpl<Type::Name, Element, Recursive>(v) {} 	\
	template<Type::Enum T> \
	explicit Name(VectorImpl<T, Element, Recursive> const& other) : VectorImpl<Type::Name, Element, Recursive>(other) {} \
	static Name c() { Name c(0); return c; } \
	static Name c(Element v0) { Name c(1); c[0] = v0; return c; } \
	static Name c(Element v0, Element v1) { Name c(2); c[0] = v0; c[1] = v1; return c; } \
	static Name c(Element v0, Element v1, Element v2) { Name c(3); c[0] = v0; c[1] = v1; c[2] = v2; return c; } \
	static Name c(Element v0, Element v1, Element v2, Element v3) { Name c(4); c[0] = v0; c[1] = v1; c[2] = v2; c[3] = v3; return c; } \
	const static bool CheckNA; 						\
	const static Element NAelement; \
	static Name NA() { static Name na = Name::c(NAelement); return na; } 
/* note missing }; */

VECTOR_IMPL(Null, unsigned char, false) 
	static Null Singleton() { static Null s = Null::c(); return s; } 
};

VECTOR_IMPL(Logical, unsigned char, false)
	static Logical True() { static Logical t = Logical::c(1); return t; }
	static Logical False() { static Logical f = Logical::c(0); return f; } 
	
	static bool isTrue(unsigned char c) { return c == 1; }
	static bool isFalse(unsigned char c) { return c == 0; }
	static bool isNA(unsigned char c) { return c == NAelement; }
	static bool isNaN(unsigned char c) { return false; }
	static bool isFinite(unsigned char c) { return false; }
	static bool isInfinite(unsigned char c) { return false; }
};

VECTOR_IMPL(Integer, int64_t, false)
	static bool isNA(int64_t c) { return c == NAelement; }
	static bool isNaN(int64_t c) { return false; }
	static bool isFinite(int64_t c) { return c != NAelement; }
	static bool isInfinite(int64_t c) { return false; }
}; 

VECTOR_IMPL(Double, double, false)
	static Double Inf() { static Double i = Double::c(std::numeric_limits<double>::infinity()); return i; }
	static Double NInf() { static Double i = Double::c(-std::numeric_limits<double>::infinity()); return i; }
	static Double NaN() { static Double n = Double::c(std::numeric_limits<double>::quiet_NaN()); return n; } 
	
	static bool isNA(double c) { _doublena a, b; a.d = c; b.d = NAelement; return a.i==b.i; }
	static bool isNaN(double c) { return (c != c) && !isNA(c); }
	static bool isFinite(double c) { return c == c && c != std::numeric_limits<double>::infinity() && c != -std::numeric_limits<double>::infinity(); }
	static bool isInfinite(double c) { return c == std::numeric_limits<double>::infinity() || c == -std::numeric_limits<double>::infinity(); }
};

VECTOR_IMPL(Complex, std::complex<double>, false)
	static bool isNA(std::complex<double> c) { _doublena a, b, t ; a.d = c.real(); b.d = c.imag(); t.d = Double::NAelement; return a.i==t.i || b.i==t.i; }
	static bool isNaN(std::complex<double> c) { return Double::isNaN(c.real()) || Double::isNaN(c.imag()); }
	static bool isFinite(std::complex<double> c) { return Double::isFinite(c.real()) && Double::isFinite(c.imag()); }
	static bool isInfinite(std::complex<double> c) { return Double::isInfinite(c.real()) || Double::isInfinite(c.imag()); }
};

VECTOR_IMPL(Character, Symbol, false)
	static bool isNA(Symbol c) { return c == Symbol::NA; }
	static bool isNaN(Symbol c) { return false; }
	static bool isFinite(Symbol c) { return false; }
	static bool isInfinite(Symbol c) { return false; }
};

VECTOR_IMPL(Raw, unsigned char, false) 
	static bool isNA(unsigned char c) { return false; }
	static bool isNaN(unsigned char c) { return false; }
	static bool isFinite(unsigned char c) { return false; }
	static bool isInfinite(unsigned char c) { return false; }
};

VECTOR_IMPL(List, Value, true) 
	static bool isNA(Value c) { return false; }
	static bool isNaN(Value c) { return false; }
	static bool isFinite(Value c) { return false; }
	static bool isInfinite(Value c) { return false; }
};

VECTOR_IMPL(PairList, Value, true) 
	static bool isNA(Value c) { return false; }
	static bool isNaN(Value c) { return false; }
	static bool isFinite(Value c) { return false; }
	static bool isInfinite(Value c) { return false; }
};

VECTOR_IMPL(Call, Value, true) 
	static bool isNA(Value c) { return false; }
	static bool isNaN(Value c) { return false; }
	static bool isFinite(Value c) { return false; }
	static bool isInfinite(Value c) { return false; }
};

VECTOR_IMPL(Expression, Value, true) 
	static bool isNA(Value c) { return false; }
	static bool isNaN(Value c) { return false; }
	static bool isFinite(Value c) { return false; }
	static bool isInfinite(Value c) { return false; }
};

inline Vector::Vector(Type::Enum t, int64_t length) {
	switch(t) {
		case Type::Null: *this = Null::Singleton(); break;
		#define CASE(Name, ...) case Type::Name: *this = Name(length); break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		default: throw RuntimeError("attempt to create invalid vector type"); break;
	};
}
inline Vector::Vector(Type::Enum t, int64_t length, void * data) {
	uint64_t width;
	switch(t) {
		#define CASE(Name,...) case Type::Name: width = Name::width; break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: throw RuntimeError("attempt to create invalid vector type"); break;
	};
	Value v;
	v.type = t;
	v.length = length;
	//CHECK: are the pointers to each member of a union the same value?
	if(width * length <= sizeof(void*))
		memcpy(&v.p,data,width * length);
	else
		v.p = data;
	v.attributes = NULL;
	*this = Vector(v);
}

inline Vector::Vector(Value v) {
	switch(v.type) {
		case Type::Null: *this = Null::Singleton(); break;
		#define CASE(Name, ...) case Type::Name: *this = Name(v); break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		default: throw RuntimeError("attempt to create invalid vector type"); break;
	};
}

struct CompiledCall : public gc {
	Call call;
	List arguments;
	int64_t dots;
	
	explicit CompiledCall(Call const& call, List const& arguments, int64_t dots) 
		: call(call), arguments(arguments), dots(dots) {}
};

struct Code : public gc {
	Value expression;
	Symbol string;

	List parameters;
	int64_t dots;
	
	std::vector<Symbol> slotSymbols;
	int64_t registers;

	std::vector<Value, traceable_allocator<Value> > constants;
	std::vector<Code*, traceable_allocator<Code*> > code; 	// constant code blocks that are nested in this code block (e.g. a nested function definition or a lazy parameter expression). Not stored in constants since it is only loaded by the 'function' op.
	std::vector<CompiledCall, traceable_allocator<CompiledCall> > calls; // nested calls

	std::vector<Instruction> bc;			// bytecode
	mutable std::vector<Instruction> tbc;		// threaded bytecode
	std::vector<Trace *> traces;
};

class Function {
private:
	Code* _code;
	Environment* _env;
public:
	explicit Function(Code* code, Environment* env)
		: _code(code), _env(env) {}
	
	explicit Function(Value const& v) : _code((Code*)v.p), _env((Environment*)v.env) {
		assert(v.type.isFunction() || v.type.isPromise());
	}

	operator Value() const {
		return Value::Make(Type::Function, 0, (void*)_code, (void*)_env);
	}

	Value AsPromise() const {
		return Value::Make(Type::Promise, 0, (void*)_code, (void*)_env);
	}

	List const& parameters() const { return _code->parameters; }
	int64_t dots() const { return _code->dots; }
	Symbol const& string() const { return _code->string; }
	
	Code* code() const { return _code; }
	Environment* environment() const { return _env; }
};

class BuiltIn {
public:
	typedef Value (*BuiltInFunctionPtr)(State& s, List const& args);
	BuiltInFunctionPtr func;
	explicit BuiltIn(BuiltInFunctionPtr func) : func(func) {}
	explicit BuiltIn(Value const& v) {
		assert(v.type.isBuiltIn());
		func = (BuiltInFunctionPtr)v.p; 
	}
	operator Value() const {
		return Value::Make(Type::BuiltIn, 0, (void*)func, 0);
	}
};

class REnvironment {
private:
	Environment* env;
public:
	Attributes* attributes;
	explicit REnvironment(Environment* env) : env(env), attributes(0) {
	}
	explicit REnvironment(Value const& v) {
		assert(v.type == Type::Environment);
		env = (Environment*)v.p;
		attributes = v.attributes;
	}
	
	operator Value() const {
		return Value::Make(Type::Environment, 0, (void*)env, attributes);
	}
	Environment* ptr() const {
		return env;
	}
};

struct Attributes : public gc {
	typedef std::map<Symbol, Value, std::less<Symbol>, gc_allocator<std::pair<Symbol, Value> > > Container;
	Container container;
};

inline bool hasAttribute(Value const& v, Symbol s) {
	return (v.attributes != 0) && v.attributes->container.find(s) != v.attributes->container.end();
}

inline bool hasNames(Value const& v) {
	return hasAttribute(v, Symbol::names);
}

inline bool hasClass(Value const& v) {
	return hasAttribute(v, Symbol::classSym);
}

inline bool hasDim(Value const& v) {
	return hasAttribute(v, Symbol::dim);
}

inline Value getAttribute(Value const& v, Symbol s) {
	if(v.attributes == 0) return Null::Singleton();
	else {
		Attributes::Container::const_iterator i = v.attributes->container.find(s);
		if(i != v.attributes->container.end()) return i->second;
		else return Null::Singleton();
	}
}

inline Value getNames(Value const& v) {
	return getAttribute(v, Symbol::names);
}

inline Value getClass(Value const& v) {
	return getAttribute(v, Symbol::classSym);
}

inline Value getDim(Value const& v) {
	return getAttribute(v, Symbol::dim);
}

template<class T, class A>
inline T setAttribute(T& v, Symbol s, A const a) {
	Attributes* attributes = new (GC) Attributes();
	if(v.attributes != 0)
		*attributes = *v.attributes;
	v.attributes = attributes;
	Value av = (Value)a;
	if(av.isNull()) {
		v.attributes->container.erase(s);
	} else {
		v.attributes->container[s] = av;
	}
	return v;
}

template<class T, class A>
inline T setNames(T& v, A const a) {
	return setAttribute(v, Symbol::names, a);
}

template<class T, class A>
inline T setClass(T& v, A const a) {
	return setAttribute(v, Symbol::classSym, a);
}

template<class T, class A>
inline T setDim(T& v, A const a) {
	return setAttribute(v, Symbol::dim, a);
}

inline bool isObject(Value const& v) {
	return hasClass(v);
}

struct Pairs {
	struct Pair {
		Symbol n;
		Value v;
	};
	std::deque<Pair, traceable_allocator<Value> > p;
	
	int64_t length() const { return p.size(); }
	void push_front(Symbol n, Value v) { Pair t = {n, v}; p.push_front(t); }
	void push_back(Symbol n, Value v)  { Pair t = {n, v}; p.push_back(t); }
	const Value& value(int64_t i) const { return p[i].v; }
	const Symbol& name(int64_t i) const { return p[i].n; }
	
	List toList(bool forceNames) const {
		List l(length());
		for(int64_t i = 0; i < length(); i++)
			l[i] = value(i);
		bool named = false;
		for(int64_t i = 0; i < length(); i++) {
			if(name(i) != Symbol::empty) {
				named = true;
				break;
			}
		}
		if(named || forceNames) {
			Character n(length());
			for(int64_t i = 0; i < length(); i++)
				n[i] = Symbol(name(i).i);
			setNames(l, n);
		}
		return l;
	}
};


/*
 * Riposte execution environments are split into two parts:
 * 1) Environment -- the static part, exposed to the R level as an R environment, these may not obey stack discipline, thus allocated on heap, try to reuse to decrease allocation overhead.
 *     -static link to enclosing environment
 *     -dynamic link to calling environment (necessary for promises which need to see original call stack)
 *     -slots storing variables that may be accessible by inner functions that may be returned 
 *		(for now, conservatively assume all variables are accessible)
 *     -names of slots
 *     -map storing overflow variables or variables that are created dynamically 
 * 		(e.g. assign() in a nested function call)
 *
 * 2) Stack Frame -- the dynamic part, not exposed at the R level, these obey stack discipline
 *     -pointer to associated Environment
 *     -pointer to Stack Frame of calling function (the dynamic link)
 *     -pointer to constant file
 *     -pointer to registers
 *     -return PC
 *     -result register pointer
 */

class Environment : public gc {
private:
	typedef std::map<Symbol, Value, std::less<Symbol>, gc_allocator<std::pair<Symbol, Value> > > Map;

	Environment* staticParent, * dynamicParent;
	Value slots[32];
	Symbol slotNames[32];		// rather than duplicating, this could be a pointer?
	unsigned char slotCount;
	Map overflow;

public:
	
	Environment(Environment* staticParent,
		    Environment* dynamicParent) : 
			staticParent(staticParent), 
			dynamicParent(dynamicParent), 
			slotCount(0) {}

	Environment(Environment* staticParent, Environment* dynamicParent, std::vector<Symbol> const& slots) : 
		staticParent(dynamicParent),
		dynamicParent(dynamicParent) {
		slotCount = slots.size();
		for(unsigned char i = 0; i < slotCount; i++)
			slotNames[i] = slots[i];
	}

 	void init(Environment* staticParent, Environment* dynamicParent, std::vector<Symbol> const& slots) {
		this->staticParent = staticParent;
		this->dynamicParent = dynamicParent;
		slotCount = slots.size();
		for(unsigned char i = 0; i < slotCount; i++)
			slotNames[i] = slots[i];
		overflow.clear();
	}

	Environment* StaticParent() const { return staticParent; }
	Environment* DynamicParent() const { return dynamicParent; }

	void setDynamicParent(Environment* env) {
		dynamicParent = env;
	}

	Value& get(int64_t i) {
		assert(i < slotCount);
		return slots[i];
	}
	Symbol slotName(int64_t i) { return slotNames[i]; }
	int SlotCount() const { return slotCount; }

	int numVariables() const { return slotCount + overflow.size(); }

	Value get(Symbol const& name) const { 
		for(uint64_t i = 0; i < slotCount; i++) {
			if(slotNames[i] == name) {
				return slots[i];
			}
		}
		Map::const_iterator i = overflow.find(name);
		if(i != overflow.end()) {
			return i->second;
		} else {
			return Nil;
		}
	}

	Value& getLocation(Symbol const& name) {
		for(uint64_t i = 0; i < slotCount; i++) {
			if(slotNames[i] == name) {
				return slots[i];
			}
		}
		Map::iterator i = overflow.find(name);
		if(i != overflow.end()) {
			return i->second;
		} else {
			throw RiposteError("variable not found in getLocation");
		}
	}

	Value getQuoted(Symbol const& name) const {
		Value value = get(name);
		if(value.isPromise()) {
			value = Function(value).code()->expression;
		}
		return value;
	}

	Function getCode(Symbol const& name) const {
		Value value = get(name);
		return Function(value);
	}

	void assign(Symbol const& name, Value const& value) {
		for(uint64_t i = 0; i < slotCount; i++) {
			if(slotNames[i] == name) {
				slots[i] = value;
				return;
			}
		}
		overflow[name] = value;
	}

	void rm(Symbol const& name) {
		for(uint64_t i = 0; i < slotCount; i++) {
			if(slotNames[i] == name) {
				slots[i] = Nil;
				return;
			}
		}
		overflow.erase(name);
	}
};

struct StackFrame {
	Environment* environment;
	bool ownEnvironment;
	Code const* code;

	Instruction const* returnpc;
	Value* returnbase;
	Value* result;
};

#define DEFAULT_NUM_REGISTERS 10000

struct State {
	Value* registers;
	Value* base;

	std::vector<StackFrame, traceable_allocator<StackFrame> > stack;
	StackFrame frame;
	std::vector<Environment*, traceable_allocator<Environment*> > environments;

	std::vector<Environment*, traceable_allocator<Environment*> > path;
	Environment* global;

	StringTable strings;
	
	std::vector<std::string> warnings;
	
	TraceState tracing; //all state related to tracing compiler

	State(Environment* global, Environment* base) {
		this->global = global;
		path.push_back(base);
		registers = new (GC) Value[DEFAULT_NUM_REGISTERS];
		this->base = registers + DEFAULT_NUM_REGISTERS;
	}

	StackFrame& push() {
		stack.push_back(frame);
		return frame;
	}

	void pop() {
		frame = stack.back();
		stack.pop_back();
	}

	//StackFrame& frame() {
	//	return stack.back();
	//}

	//StackFrame& frame(int fromBack) {
	//	return stack[stack.size()-fromBack-1];
	//}

	std::string stringify(Value const& v) const;
	std::string stringify(Trace const & t) const;
	std::string deparse(Value const& v) const;

	Symbol StrToSym(std::string s) {
		return Symbol(strings.in(s));
	}

	std::string const& SymToStr(Symbol s) const {
		return strings.out(s.i);
	}
};



Value eval(State& state, Function const& function);
Value eval(State& state, Code const* code, Environment* environment); 
Value eval(State& state, Code const* code);
void interpreter_init(State& state);

#endif
