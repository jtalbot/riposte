
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
#define GC_DEBUG
#include <gc/gc_cpp.h>
#include <gc/gc_allocator.h>

#include "type.h"
#include "bc.h"
#include "common.h"
#include "enum.h"
#include "symbols.h"
#include "exceptions.h"
#include "trace.h"

struct Value {
	
	union {
		struct {
			Type::Enum type:4;
			//unsigned char flags:0;
			int64_t length:60;
		};
		int64_t header;
	};
	union {
		void* p;
		int64_t i;
		double d;
		unsigned char c;
	};

	static Value Make(Type::Enum type, int64_t length, int64_t data) {
		Value v = {{{type, length}}, {(void*)data}};
		return v;
	}

	static Value Make(Type::Enum type, int64_t length, void* data) {
		Value v = {{{type, length}}, {data}};
		return v;
	}

	static void Init(Value& v, Type::Enum type, int64_t length) {
		v.header =  type + (length<<4);
	}

	bool isNil() const { return type == Type::Nil; }
	bool isNull() const { return type == Type::Null; }
	bool isLogical() const { return type == Type::Logical; }
	bool isInteger() const { return type == Type::Integer; }
	bool isDouble() const { return type == Type::Double; }
	bool isComplex() const { return type == Type::Complex; }
	bool isCharacter() const { return type == Type::Character; }
	bool isList() const { return type == Type::List; }
	bool isSymbol() const { return type == Type::Symbol; }
	bool isPromise() const { return type == Type::Promise; }
	bool isFunction() const { return type == Type::Function; }
	bool isBuiltIn() const { return type == Type::BuiltIn; }
	bool isObject() const { return type == Type::Object; }
	bool isMathCoerce() const { return isDouble() || isInteger() || isLogical() || isComplex(); }
	bool isLogicalCoerce() const { return isDouble() || isInteger() || isLogical() || isComplex(); }
	bool isVector() const { return isNull() || isLogical() || isInteger() || isDouble() || isComplex() || isCharacter() || isList(); }
	bool isClosureSafe() const { return isNull() || isLogical() || isInteger() || isDouble() || isComplex() || isCharacter() || isSymbol() || (isList() && length==0); }
	bool isConcrete() const { return type < Type::Promise; }

	template<class T> T& scalar() { throw "not allowed"; }
	template<class T> T const& scalar() const { throw "not allowed"; }

	static Value const& Nil() { static const Value v = { {{Type::Nil, 0}}, {0} }; return v; }
};

class Environment;
class State;

// Value type implementations

struct Symbol {
	int64_t i;

	Symbol() : i(1) {} // Defaults to the empty symbol...
	explicit Symbol(int64_t index) : i(index) {}

	explicit Symbol(Value const& v) : i(v.i) {
		assert(v.type == Type::Symbol); 
	}

	operator Value() const {
		return Value::Make(Type::Symbol, 0, i);
	}

	bool operator<(Symbol const& other) const { return i < other.i;	}
	bool operator==(Symbol const& other) const { return i == other.i; }
	bool operator!=(Symbol const& other) const { return i != other.i; }

	bool operator==(int64_t other) const { return i == other; }
};

// make constant members matching built in strings
namespace Symbols {
	#define CONST_DECLARE(name, string, ...) static const ::Symbol name(String::name);
	STRINGS(CONST_DECLARE)
	#undef CONST_DECLARE
}

template<> inline int64_t& Value::scalar<int64_t>() { return i; }
template<> inline double& Value::scalar<double>() { return d; }
template<> inline unsigned char& Value::scalar<unsigned char>() { return c; }
template<> inline Symbol& Value::scalar<Symbol>() { return (Symbol&)i; }

template<> inline int64_t const& Value::scalar<int64_t>() const { return i; }
template<> inline double const& Value::scalar<double>() const { return d; }
template<> inline unsigned char const& Value::scalar<unsigned char>() const { return c; }
template<> inline Symbol const& Value::scalar<Symbol>() const { return (Symbol const&) i; }


template<Type::Enum VType, typename ElementType, bool Recursive,
	bool canPack = sizeof(ElementType) <= sizeof(int64_t) && !Recursive>
struct Vector : public Value {
	typedef ElementType Element;
	static const int64_t width = sizeof(ElementType); 
	static const Type::Enum VectorType = VType;

	bool isScalar() const {
		return length == 1;
	}

	ElementType& s() { return canPack ? Value::scalar<ElementType>() : *(ElementType*)p; }
	ElementType const& s() const { return canPack ? Value::scalar<ElementType>() : *(ElementType const*)p; }

	ElementType const* v() const { return (canPack && isScalar()) ? &Value::scalar<ElementType>() : (ElementType const*)p; }
	ElementType* v() { return (canPack && isScalar()) ? &Value::scalar<ElementType>() : (ElementType*)p; }
	
	ElementType& operator[](int64_t index) { return v()[index]; }
	ElementType const& operator[](int64_t index) const { return v()[index]; }

	explicit Vector(int64_t length=0) {
		Value::Init(*this, VectorType, length);
		if(canPack && length > 1)
			p = Recursive ? new (GC) Element[length] : 
				new (PointerFreeGC) Element[length];
		else if(!canPack && length > 0)
			p = Recursive ? new (GC) Element[length] : 
				new (PointerFreeGC) Element[length];
	}

	static Vector<VType, ElementType, Recursive>& Init(Value& v, int64_t length) {
		Value::Init(v, VectorType, length);
		if(canPack && length > 1)
			v.p = Recursive ? new (GC) Element[length] : 
				new (PointerFreeGC) Element[length];
		else if(!canPack && length > 0)
			v.p = Recursive ? new (GC) Element[length] : 
				new (PointerFreeGC) Element[length];
		return (Vector<VType, ElementType, Recursive>&)v;
	}

	static void InitScalar(Value& v, ElementType const& d) {
		Value::Init(v, VectorType, 1);
		if(canPack)
			v.scalar<ElementType>() = d;
		else {
			v.p = Recursive ? new (GC) Element[1] : new (PointerFreeGC) Element[1];
			*(Element*)v.p = d;
		}
	}

	explicit Vector(Value const& v) {
		assert(v.type == VType);
		type = VType;
		length = v.length;
		p = v.p;
	}

	operator Value() const {
		Value v;
		return Value::Init(type, length, p);
		return v;
	}
};

union _doublena {
	int64_t i;
	double d;
};

#define VECTOR_IMPL(Name, Element, Recursive) 				\
struct Name : public Vector<Type::Name, Element, Recursive> { 			\
	explicit Name(int64_t length=0) : Vector<Type::Name, Element, Recursive>(length) {} 	\
	explicit Name(Value const& v) : Vector<Type::Name, Element, Recursive>(v) {} 	\
	template<Type::Enum T> \
	explicit Name(Vector<T, Element, Recursive> const& other) : Vector<Type::Name, Element, Recursive>(other) {} \
	static Name c() { Name c(0); return c; } \
	static Name c(Element v0) { Name c(1); c[0] = v0; return c; } \
	static Name c(Element v0, Element v1) { Name c(2); c[0] = v0; c[1] = v1; return c; } \
	static Name c(Element v0, Element v1, Element v2) { Name c(3); c[0] = v0; c[1] = v1; c[2] = v2; return c; } \
	static Name c(Element v0, Element v1, Element v2, Element v3) { Name c(4); c[0] = v0; c[1] = v1; c[2] = v2; c[3] = v3; return c; } \
	const static Element NAelement; \
	static Name NA() { static Name na = Name::c(NAelement); return na; }  \
	static Name& Init(Value& v, int64_t length) { return (Name&)Vector<Type::Name, Element, Recursive>::Init(v, length); } \
	static void InitScalar(Value& v, Element const& d) { Vector<Type::Name, Element, Recursive>::InitScalar(v, d); }
/* note missing }; */

VECTOR_IMPL(Null, unsigned char, false) 
	static const bool CheckNA = false;
	static Null Singleton() { static Null s = Null::c(); return s; } 
};

VECTOR_IMPL(Logical, unsigned char, false)
	static const bool CheckNA = true;
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
	static const bool CheckNA = true;
	static bool isNA(int64_t c) { return c == NAelement; }
	static bool isNaN(int64_t c) { return false; }
	static bool isFinite(int64_t c) { return c != NAelement; }
	static bool isInfinite(int64_t c) { return false; }
}; 

VECTOR_IMPL(Double, double, false)
	static const bool CheckNA = false;
	static Double Inf() { static Double i = Double::c(std::numeric_limits<double>::infinity()); return i; }
	static Double NInf() { static Double i = Double::c(-std::numeric_limits<double>::infinity()); return i; }
	static Double NaN() { static Double n = Double::c(std::numeric_limits<double>::quiet_NaN()); return n; } 
	
	static bool isNA(double c) { _doublena a, b; a.d = c; b.d = NAelement; return a.i==b.i; }
	static bool isNaN(double c) { return (c != c) && !isNA(c); }
	static bool isFinite(double c) { return c == c && c != std::numeric_limits<double>::infinity() && c != -std::numeric_limits<double>::infinity(); }
	static bool isInfinite(double c) { return c == std::numeric_limits<double>::infinity() || c == -std::numeric_limits<double>::infinity(); }
};

VECTOR_IMPL(Complex, std::complex<double>, false)
	static const bool CheckNA = false;
	static bool isNA(std::complex<double> c) { _doublena a, b, t ; a.d = c.real(); b.d = c.imag(); t.d = Double::NAelement; return a.i==t.i || b.i==t.i; }
	static bool isNaN(std::complex<double> c) { return Double::isNaN(c.real()) || Double::isNaN(c.imag()); }
	static bool isFinite(std::complex<double> c) { return Double::isFinite(c.real()) && Double::isFinite(c.imag()); }
	static bool isInfinite(std::complex<double> c) { return Double::isInfinite(c.real()) || Double::isInfinite(c.imag()); }
};

VECTOR_IMPL(Character, Symbol, false)
	static const bool CheckNA = true;
	static bool isNA(Symbol c) { return c == Symbols::NA; }
	static bool isNaN(Symbol c) { return false; }
	static bool isFinite(Symbol c) { return false; }
	static bool isInfinite(Symbol c) { return false; }
};

VECTOR_IMPL(Raw, unsigned char, false) 
	static const bool CheckNA = false;
	static bool isNA(unsigned char c) { return false; }
	static bool isNaN(unsigned char c) { return false; }
	static bool isFinite(unsigned char c) { return false; }
	static bool isInfinite(unsigned char c) { return false; }
};

VECTOR_IMPL(List, Value, true) 
	static const bool CheckNA = false;
	static bool isNA(Value c) { return false; }
	static bool isNaN(Value c) { return false; }
	static bool isFinite(Value c) { return false; }
	static bool isInfinite(Value c) { return false; }
};

struct CompiledCall : public gc {
	List call;

	List arguments;
	Character names;
	int64_t dots;
	
	explicit CompiledCall(List const& call, List const& arguments, Character const& names, int64_t dots) 
		: call(call), arguments(arguments), names(names), dots(dots) {}
};

struct Prototype : public gc {
	Value expression;
	Symbol string;

	List parameters;
	Character names;
	
	std::vector<Symbol> slotSymbols;
	int dots;
	int registers;

	std::vector<Value, traceable_allocator<Value> > constants;
	std::vector<Prototype*, traceable_allocator<Prototype*> > prototypes; 	
	std::vector<CompiledCall, traceable_allocator<CompiledCall> > calls; 

	std::vector<Instruction> bc;			// bytecode
	mutable std::vector<Instruction> tbc;		// threaded bytecode
	std::vector<Trace *> traces;
};

class Function {
private:
	Prototype* proto;
	Environment* env;
public:
	explicit Function(Prototype* proto, Environment* env)
		: proto(proto), env(env) {}
	
	explicit Function(Value const& v) {
		assert(v.isFunction() || v.isPromise());
		proto = (Prototype*)(v.length << 4);
		env = (Environment*)v.p;
	}

	operator Value() const {
		Value v;
		v.header = (int64_t)proto + Type::Function;
		v.p = env;
		return v;
		//return Value::Make(Type::Function, 0, (void*)proto, (void*)env);
	}

	Value AsPromise() const {
		Value v;
		v.header = (int64_t)proto + Type::Promise;
		v.p = env;
		return v;
		//return Value::Make(Type::Promise, 0, (void*)proto, (void*)env);
	}

	List const& parameters() const { return proto->parameters; }
	Character const& names() const { return proto->names; }
	int64_t dots() const { return proto->dots; }
	Symbol const& string() const { return proto->string; }
	
	Prototype* prototype() const { return proto; }
	Environment* environment() const { return env; }
};

class BuiltIn {
public:
	typedef Value (*BuiltInFunctionPtr)(State& s, List const& args, Character const& names);
	BuiltInFunctionPtr func;
	explicit BuiltIn(BuiltInFunctionPtr func) : func(func) {}
	explicit BuiltIn(Value const& v) {
		assert(v.isBuiltIn());
		func = (BuiltInFunctionPtr)v.p; 
	}
	operator Value() const {
		return Value::Make(Type::BuiltIn, 0, (void*)func);
	}
};

class REnvironment {
private:
	Environment* env;
public:
	explicit REnvironment(Environment* env) : env(env) {
	}
	explicit REnvironment(Value const& v) {
		assert(v.type == Type::Environment);
		env = (Environment*)v.p;
	}
	
	operator Value() const {
		Value v;
		Value::Init(v, Type::Environment, 0);
		v.p = (void*)env;
		return v;
	}
	Environment* ptr() const {
		return env;
	}
};

struct Object : public Value {
	typedef std::map<Symbol, Value, std::less<Symbol>, traceable_allocator<std::pair<Symbol, Value> > > Container;

	struct Inner : public gc {
		Value base;
		Container attributes;
	};

	Value& base() { return ((Inner*)p)->base; }
	Value const& base() const { return ((Inner*)p)->base; }
	Container& attributes() { return ((Inner*)p)->attributes; }
	Container const& attributes() const { return ((Inner*)p)->attributes; }

	static void Init(Value& v, Value const& _base) {
		assert(_base.isList());
		Value::Init(v, Type::Object, 0);
		Inner* inner = new Inner();
		inner->base = _base;
		v.p = (void*)inner;
	}

	static void InitWithNames(Value& v, Value const& _base, Value const& _names) {
		Init(v, _base);
		if(!_names.isNil())
			((Object&)v).setNames(_names);
	}
	
	static void Init(Value& v, Value const& _base, Value const& _names, Value const& className) {
		InitWithNames(v, _base, _names);
		((Object&)v).setClass(className);
	}

	bool hasAttribute(Symbol s) const {
		return attributes().find(s) != attributes().end();
	}

	bool hasNames() const { return hasAttribute(Symbols::names); }
	bool hasClass() const { return hasAttribute(Symbols::classSym); }
	bool hasDim() const   { return hasAttribute(Symbols::dim); }

	Value getAttribute(Symbol s) const {
		Container::const_iterator i = attributes().find(s);
		if(i != attributes().end()) return i->second;
		else return Value::Nil();
	}

	Value getNames() const { return getAttribute(Symbols::names); }
	Value getClass() const { return getAttribute(Symbols::classSym); }
	Value getDim() const { return getAttribute(Symbols::dim); }

	void setAttribute(Symbol s, Value const& v) {
		attributes()[s] = v;
	}
	
	void setNames(Value const& v) { setAttribute(Symbols::names, v); }
	void setClass(Value const& v) { setAttribute(Symbols::classSym, v); }
	void setDim(Value const& v)  { setAttribute(Symbols::dim, v); }

	Symbol className() const {
		if(!hasClass()) {
			return Symbol(base().type);
		}
		else {
			return Character(getClass())[0];
		}
	}
};

inline Value CreateExpression(List const& list) {
	Value v;
	Object::Init(v, list, Value::Nil(), Character::c(Symbols::Expression));
	return v;
}

inline Value CreateCall(List const& list, Value const& names = Value::Nil()) {
	Value v;
	Object::Init(v, list, names, Character::c(Symbols::Call));
	return v;
}

struct Pairs {
	struct Pair {
		Symbol n;
		Value v;
	};
	std::deque<Pair, traceable_allocator<Value> > p;
	
	int64_t length() const { return p.size(); }
	void push_front(Symbol n, Value const& v) { Pair t = {n, v}; p.push_front(t); }
	void push_back(Symbol n, Value const& v)  { Pair t = {n, v}; p.push_back(t); }
	const Value& value(int64_t i) const { return p[i].v; }
	const Symbol& name(int64_t i) const { return p[i].n; }

	List values() const {
		List l(length());
		for(int64_t i = 0; i < length(); i++)
			l[i] = value(i);
		return l;
	}

	Value names(bool forceNames) const {
		bool named = false;
		for(int64_t i = 0; i < length(); i++) {
			if(name(i) != Symbols::empty) {
				named = true;
				break;
			}
		}
		if(named || forceNames) {
			Character n(length());
			for(int64_t i = 0; i < length(); i++)
				n[i] = Symbol(name(i).i);
			return n;
		}
		else return Value::Nil();
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
	typedef std::map<Symbol, Value, std::less<Symbol>, traceable_allocator<std::pair<Symbol, Value> > > Map;

	Value slots[32];
	Symbol slotNames[32];		// rather than duplicating, this could be a pointer?
	Environment* staticParent, * dynamicParent;
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

	Value const& get(Symbol const& name) const { 
		for(uint64_t i = 0; i < slotCount; i++) {
			if(slotNames[i] == name) {
				return slots[i];
			}
		}
		Map::const_iterator i = overflow.find(name);
		if(i != overflow.end()) {
			return i->second;
		} else {
			return Value::Nil();
		}
	}

	Value const& hget(Symbol const& name) const { 
		Map::const_iterator i = overflow.find(name);
		if(i != overflow.end()) {
			return i->second;
		} else {
			return Value::Nil();
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
			value = Function(value).prototype()->expression;
		}
		return value;
	}

	Function getCode(Symbol const& name) const {
		Value value = get(name);
		return Function(value);
	}

	void hassign(Symbol const& name, Value const& value) {
		overflow[name] = value;
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
				slots[i] = Value::Nil();
				return;
			}
		}
		overflow.erase(name);
	}
};

struct StackFrame {
	Environment* environment;
	bool ownEnvironment;
	Prototype const* prototype;

	Instruction const* returnpc;
	Value* returnbase;
	Value* result;
};

#define DEFAULT_NUM_REGISTERS 10000

struct State {
	Value* base;
	Value* registers;

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
Value eval(State& state, Prototype const* prototype, Environment* environment); 
Value eval(State& state, Prototype const* prototype);
void interpreter_init(State& state);

#endif
