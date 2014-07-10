
#ifndef _RIPOSTE_VALUE_H
#define _RIPOSTE_VALUE_H

#include <vector>
#include <assert.h>

#include "common.h"
#include "type.h"
#include "gc.h"
#include "strings.h"
#include "exceptions.h"

// Forward declare the HeapObjects
class State;
class Thread;
struct Code;
struct Prototype;
class Environment;
class Dictionary;
class Trace;
class Context;


struct Value {
	
	union {
		uint64_t header;
		struct {
			Type::Enum typ:8;
			uint8_t pac;
		};
	};
	union {
		void* p;
		int64_t i;
		double d;
		char c;
        unsigned char u;
		String s;
	};

	static void Init(Value& v, Type::Enum type, int64_t packed) {
		v.header = type + (packed<<8);
	}

	// Warning: shallow equality!
	bool operator==(Value const& other) const {
		return header == other.header && p == other.p;
	}
	
	bool operator!=(Value const& other) const {
		return header != other.header || p != other.p;
	}

	Type::Enum type() const { return (Type::Enum)typ; }
	uint64_t packed() const { return pac; }

	bool isNil() const 	{ return type() == Type::Nil; }
	bool isPromise() const 	{ return type() == Type::Promise; }
	bool isObject() const 	{ return type() > Type::Promise; }
	
	bool isFuture() const 	{ return type() == Type::Future; }
	bool isClosure() const { return type() == Type::Closure; }
	bool isEnvironment() const { return type() == Type::Environment; }
	
	bool isNull() const 	{ return type() == Type::Null; }
	bool isRaw() const 	{ return type() == Type::Raw; }
	bool isLogical() const 	{ return type() == Type::Logical; }
	bool isInteger() const 	{ return type() == Type::Integer; }
	bool isDouble() const 	{ return type() == Type::Double; }
	bool isCharacter() const { return type() == Type::Character; }
	bool isList() const 	{ return type() == Type::List; }
	
	bool isRaw1() const 	{ return header == Type::Raw+(1<<8); }
	bool isLogical1() const { return header == Type::Logical+(1<<8); }
	bool isInteger1() const { return header == Type::Integer+(1<<8); }
	bool isDouble1() const 	{ return header == Type::Double+(1<<8); }
	bool isCharacter1() const { return header == Type::Character+(1<<8); }
	
	bool isMathCoerce() const { return type() >= Type::Logical && type() <= Type::Double; }
	bool isLogicalCoerce() const { return type() >= Type::Logical && type() <= Type::Double; }
	bool isAtomic() const { return type() >= Type::Null && type() < Type::List; }
	bool isVector() const { return type() >= Type::Null && type() <= Type::List; }
	
	static Value const& Nil() { static const Value v = {{0}, {0}}; return v; }
};

//
// Value type implementations
//

// Promises

struct Promise : public Value {
	static const Type::Enum ValueType = Type::Promise;

    enum PromiseType {
        Default=(1<<0),

		Expression=(1<<1),
		Dots=(1<<2)
	};

	static Promise& Init(Value& v, Environment* env, Code* code, bool isDefault) {
		Value::Init(v, Type::Promise, Expression|isDefault);
		v.header += (((uint64_t)env) << 16);
		v.p = code;
		return (Promise&)v;
	}

	static Promise& Init(Value& v, Environment* env, uint64_t dotindex, bool isDefault) {
		Value::Init(v, Type::Promise, Dots|isDefault);
		v.header += (((uint64_t)env) << 16);
		v.i = dotindex;
		return (Promise&)v;
	}

    bool isDefault() const { return packed() & Default; }
	bool isExpression() const { return packed() & Expression; }
	bool isDotdot() const { return packed() & Dots; }

	Environment* environment() const { 
		return (Environment*)(((uint64_t)header) >> 16); 
	}

	Code* code() const { 
		return (Code*)p; 
	}

	uint64_t dotIndex() const {
		return i;
	}
	
	void environment(Environment* env) {
		header = (((uint64_t)env) << 16) + (pac << 8) + Type::Promise;
	}
};


// Objects

struct Object : public Value {
	Dictionary* attributes() const { 
		return (Dictionary*)(header >> 16); 
	}
	
	void attributes(Dictionary* d) {
		header = (header & ((1<<16)-1)) | ((uint64_t)d << 16);
	}

	bool hasAttributes() const { 
		return attributes() != 0; 
	}
};

typedef int16_t IRef;
struct Future : public Object {
	static const Type::Enum ValueType = Type::Future;
	static Future& Init(Value& f, Trace* trace, IRef ref) {
		Value::Init(f,Type::Future,0);
		f.i = (((uint64_t)trace) << 16) + ref;
		return (Future&)f;
	}

	Trace* trace() const { return (Trace*)(((uint64_t)i)>>16); }
	IRef ref() const { return (IRef)(uint64_t)i; }
};

struct Closure : public Object {
	static const Type::Enum ValueType = Type::Closure;
		
	struct Inner : public HeapObject {
		Prototype const* proto;
		Environment* env;
		Inner(Prototype const* proto, Environment* env)
			: proto(proto), env(env) {}
	};

	static Closure& Init(Value& v, Prototype const* proto, Environment* env) {
		Value::Init(v, Type::Closure, 0);
		v.p = new Inner(proto, env);
		return (Closure&)v;
	}

	Prototype const* prototype() const { return ((Inner*)p)->proto; }
	Environment* environment() const { return ((Inner*)p)->env; }
};

struct REnvironment : public Object {
	static const Type::Enum ValueType = Type::Environment;
	
	static REnvironment& Init(Value& v, Environment* env) {
		Value::Init(v, Type::Environment, 0);
		v.p = (void*)env;
		return (REnvironment&)v;
	}
	
	Environment* environment() const {
		return (Environment*)p;
	}
};

struct Externalptr : public Object {
    typedef void (*Finalizer)(Value v);

    struct Inner : public HeapObject {
        void* ptr;
        Value tag;
        Value prot;
        Finalizer fun;
        Inner(void* ptr, Value const& tag, Value const& prot, Finalizer fun)
            : ptr(ptr), tag(tag), prot(prot), fun(fun) {}
    };

    static void Finalize(HeapObject* o) {
        Inner* i = (Inner*)o;
        Value v;
        Value::Init(v, Type::Externalptr, 0);
        v.p = i;
        i->fun(v);
    }

    static Externalptr& Init(Value& v, void* ptr, Value tag, Value prot, Finalizer fun) {
        Value::Init(v, Type::Externalptr, 0);
        v.p = new (Finalize) Inner(ptr, tag, prot, fun);
        return (Externalptr&)v;
    }

    void* ptr() const { return ((Inner*)p)->ptr; }
    Value tag() const { return ((Inner*)p)->tag; }
    Value prot() const { return ((Inner*)p)->prot; }
    Finalizer fun() const { return ((Inner*)p)->fun; }
};

struct Vector : public Object {
	struct Inner : public HeapObject {
		int64_t length;
		int64_t capacity;
	};

	int64_t length() const { return packed() <= 1 ? (int64_t)packed() : ((Inner*)p)->length; }
	bool isScalar() const { return length() == 1; }
	void* raw() { return (void*)(((Inner*)p)+1); }	// assumes that data is immediately after capacity
	void const* raw() const { return (void const*)(((Inner*)p)+1); }	// assumes that data is immediately after capacity
	
    template<class T> T& scalar() { throw "not allowed"; }
	template<class T> T const& scalar() const { throw "not allowed"; }
};

template<> inline int64_t& Vector::scalar<int64_t>() { return i; }
template<> inline double& Vector::scalar<double>() { return d; }
template<> inline char& Vector::scalar<char>() { return c; }
template<> inline String& Vector::scalar<String>() { return s; }
template<> inline unsigned char& Vector::scalar<unsigned char>() { return u; }

template<> inline int64_t const& Vector::scalar<int64_t>() const { return i; }
template<> inline double const& Vector::scalar<double>() const { return d; }
template<> inline char const& Vector::scalar<char>() const { return c; }
template<> inline String const& Vector::scalar<String>() const { return s; }
template<> inline unsigned char const& Vector::scalar<unsigned char>() const { return u; }



template<Type::Enum VType, typename ElementType, bool Recursive>
struct VectorImpl : public Vector {

	typedef ElementType Element;
	static const Type::Enum ValueType = VType;
	static const bool canPack = sizeof(ElementType) <= sizeof(int64_t) && !Recursive;

	struct Inner : public Vector::Inner {
		ElementType data[];
	};

	ElementType const* v() const { 
		return (canPack && packed()==1) ? 
			&scalar<ElementType>() : ((Inner*)p)->data; 
	}
	ElementType* v() { 
		return (canPack && packed()==1) ? 
			&scalar<ElementType>() : ((Inner*)p)->data; 
	}

	Inner* inner() const {
		return (length() > canPack) ? (Inner*)p : 0;
	}
	
	ElementType& operator[](int64_t index) { return v()[index]; }
	ElementType const& operator[](int64_t index) const { return v()[index]; }

	static VectorImpl<VType, ElementType, Recursive>& Init(Value& v, int64_t length) {
		if((canPack && length > 1) || (!canPack && length > 0)) {
			Value::Init(v, ValueType, 2);
			int64_t l = length;
			// round l up to nearest even number so SSE can work on tail region
			l += (int64_t)((uint64_t)l & 1);
			int64_t length_aligned = (l < 128) ? (l + 1) : l;
			
			Inner* i = new (sizeof(Element)*length_aligned) Inner();
			i->length = length;
			i->capacity = length_aligned; 
			v.p = (void*)i;
		} else {
			Value::Init(v, ValueType, length);
			v.p = 0;
		}
		return (VectorImpl<VType, ElementType, Recursive>&)v;
	}

	static void InitScalar(Value& _v, ElementType const& d) {
        Vector& v = (Vector&) _v;
		Value::Init(v, ValueType, 1);
		if(canPack)
			v.scalar<ElementType>() = d;
		else {
			Inner* i = new (sizeof(Element)*2) Inner();
			i->length = 1;
			i->capacity = 1;
			i->data[0] = d;
			v.p = (void*)i;
		}
	}
};

#define VECTOR_IMPL(Name, Element, Recursive) 				\
struct Name : public VectorImpl<Type::Name, Element, Recursive> { 			\
	explicit Name(int64_t length=0) { Init(*this, length); } 	\
	static Name c() { Name c(0); return c; } \
	static Name c(Element v0) { Name c(1); c[0] = v0; return c; } \
	static Name c(Element v0, Element v1) { Name c(2); c[0] = v0; c[1] = v1; return c; } \
	static Name c(Element v0, Element v1, Element v2) { Name c(3); c[0] = v0; c[1] = v1; c[2] = v2; return c; } \
	static Name c(Element v0, Element v1, Element v2, Element v3) { Name c(4); c[0] = v0; c[1] = v1; c[2] = v2; c[3] = v3; return c; } \
	const static Element NAelement; \
	static Name NA() { static Name na = Name::c(NAelement); return na; }  \
	static Name& Init(Value& v, int64_t length) { return (Name&)VectorImpl<Type::Name, Element, Recursive>::Init(v, length); } \
	static void InitScalar(Value& v, Element const& d) { VectorImpl<Type::Name, Element, Recursive>::InitScalar(v, d); \
}\
/* note missing }; */

VECTOR_IMPL(Null, unsigned char, false)  
	static Null const& Singleton() { static Null s = Null::c(); return s; } 
	static bool isNA() { return false; }
	static bool isCheckedNA() { return false; }
};

VECTOR_IMPL(Logical, char, false)
	const static char TrueElement;
	const static char FalseElement;

	static Logical const& True() { static Logical t = Logical::c(-1); return t; }
	static Logical const& False() { static Logical f = Logical::c(0); return f; } 
	
	static bool isTrue(char c) { return c == -1; }
	static bool isFalse(char c) { return c == 0; }
	static bool isNA(char c) { return c == 1; }
	static bool isCheckedNA(char c) { return isNA(c); }
};

VECTOR_IMPL(Integer, int64_t, false)
	static bool isNA(int64_t c) { return c == NAelement; }
	static bool isCheckedNA(int64_t c) { return isNA(c); }
}; 

VECTOR_IMPL(Double, double, false)
	union _doublena {
		int64_t i;
		double d;
	};

	static Double const& Inf() { static Double i = Double::c(std::numeric_limits<double>::infinity()); return i; }
	static Double const& NInf() { static Double i = Double::c(-std::numeric_limits<double>::infinity()); return i; }
	static Double const& NaN() { static Double n = Double::c(std::numeric_limits<double>::quiet_NaN()); return n; } 
	
	static bool isNA(double c) { _doublena a, b; a.d = c; b.d = NAelement; return a.i==b.i; }
	static bool isNaN(double c) { return (c != c) && !isNA(c); }
	static bool isCheckedNA(int64_t c) { return false; }
};

VECTOR_IMPL(Character, String, false)
	static bool isNA(String c) { return c == Strings::NA; }
	static bool isCheckedNA(String c) { return isNA(c); }
};

VECTOR_IMPL(Raw, unsigned char, false) 
	static bool isNA(unsigned char c) { return false; }
	static bool isCheckedNA(unsigned char c) { return false; }
};

VECTOR_IMPL(List, Value, true) 
	static bool isNA(Value const& c) { return c.isNil(); }
	static bool isCheckedNA(Value const& c) { return isNA(c); }
};

#endif

