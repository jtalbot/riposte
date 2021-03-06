
#ifndef _RIPOSTE_VALUE_H
#define _RIPOSTE_VALUE_H

#include <vector>
#include <assert.h>

#include "common.h"
#include "type.h"
#include "bc.h"
#include "gc.h"
#include "strings.h"
#include "exceptions.h"

typedef int16_t IRef;

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
	bool isFunction() const { return type() == Type::Function; }
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
	bool isVector() const { return type() >= Type::Null && type() <= Type::List; }
	
	template<class T> T& scalar() { throw "not allowed"; }
	template<class T> T const& scalar() const { throw "not allowed"; }

	static Value const& Nil() { static const Value v = {{0}, {0}}; return v; }
};

template<> inline int64_t& Value::scalar<int64_t>() { return i; }
template<> inline double& Value::scalar<double>() { return d; }
template<> inline char& Value::scalar<char>() { return c; }
template<> inline String& Value::scalar<String>() { return s; }

template<> inline int64_t const& Value::scalar<int64_t>() const { return i; }
template<> inline double const& Value::scalar<double>() const { return d; }
template<> inline char const& Value::scalar<char>() const { return c; }
template<> inline String const& Value::scalar<String>() const { return s; }


// Name-value Pairs are used throughout the code...
struct Pair { String n; Value v; };
// Not the same as the publically visible PairList which is just an S3 class
typedef std::vector<Pair> PairList;

//
// Value type implementations
//
class State;
class Thread;
struct Prototype;
class Environment;
class Dictionary;
class Trace;


// Promises

struct Promise : public Value {
	enum PromiseType {
		NIL = 0,
		PROTOTYPE = 1,
		PROTOTYPE_DEFAULT = 2,
		DOTDOT = 3,
		DOTDOT_DEFAULT = 4
	};
	static const Type::Enum ValueType = Type::Promise;
	static Promise& Init(Value& v, Environment* env, Prototype* proto, bool isDefault) {
		Value::Init(v, Type::Promise, isDefault ? PROTOTYPE_DEFAULT : PROTOTYPE);
		v.header += (((uint64_t)env) << 16);
		v.p = proto;
		return (Promise&)v;
	}
	static Promise& Init(Value& v, Environment* env, uint64_t dotindex, bool isDefault) {
		Value::Init(v, Type::Promise, isDefault ? DOTDOT_DEFAULT : DOTDOT);
		v.header += (((uint64_t)env) << 16);
		v.i = dotindex;
		return (Promise&)v;
	}

	bool isDefault() const { 
		return packed() == PROTOTYPE_DEFAULT || packed() == DOTDOT_DEFAULT; 
	}
	bool isPrototype() const {
		return packed() == PROTOTYPE || packed() == PROTOTYPE_DEFAULT;
	}
	bool isDotdot() const {
		return packed() == DOTDOT || packed() == DOTDOT_DEFAULT;
	}
	Environment* environment() const { 
		return (Environment*)(((uint64_t)header) >> 16); 
	}
	Prototype* prototype() const { 
		return (Prototype*)p; 
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

struct Function : public Object {
	static const Type::Enum ValueType = Type::Function;
		
	struct Inner : public HeapObject {
		Prototype* proto;
		Environment* env;
		Inner(Prototype* proto, Environment* env)
			: proto(proto), env(env) {}
	};

	static Function& Init(Value& v, Prototype* proto, Environment* env) {
		Value::Init(v, Type::Function, 0);
		v.p = new Inner(proto, env);
		return (Function&)v;
	}

	Prototype* prototype() const { return ((Inner*)p)->proto; }
	Environment* environment() const { return ((Inner*)p)->env; }
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
};

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
			&Value::scalar<ElementType>() : ((Inner*)p)->data; 
	}
	ElementType* v() { 
		return (canPack && packed()==1) ? 
			&Value::scalar<ElementType>() : ((Inner*)p)->data; 
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

	static void InitScalar(Value& v, ElementType const& d) {
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
	static bool isNaN(char c) { return false; }
	static bool isFinite(char c) { return false; }
	static bool isInfinite(char c) { return false; }
};

VECTOR_IMPL(Integer, int64_t, false)
	static bool isNA(int64_t c) { return c == NAelement; }
	static bool isCheckedNA(int64_t c) { return isNA(c); }
	static bool isNaN(int64_t c) { return false; }
	static bool isFinite(int64_t c) { return c != NAelement; }
	static bool isInfinite(int64_t c) { return false; }
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
	static bool isCheckedNA(int64_t c) { return false; }
	static bool isNaN(double c) { return (c != c) && !isNA(c); }
	static bool isFinite(double c) { return c == c && c != std::numeric_limits<double>::infinity() && c != -std::numeric_limits<double>::infinity(); }
	static bool isInfinite(double c) { return c == std::numeric_limits<double>::infinity() || c == -std::numeric_limits<double>::infinity(); }
};

VECTOR_IMPL(Character, String, false)
	static bool isNA(String c) { return c == Strings::NA; }
	static bool isCheckedNA(String c) { return isNA(c); }
	static bool isNaN(String c) { return false; }
	static bool isFinite(String c) { return false; }
	static bool isInfinite(String c) { return false; }
};

VECTOR_IMPL(Raw, unsigned char, false) 
	static bool isNA(unsigned char c) { return false; }
	static bool isCheckedNA(unsigned char c) { return false; }
	static bool isNaN(unsigned char c) { return false; }
	static bool isFinite(unsigned char c) { return false; }
	static bool isInfinite(unsigned char c) { return false; }
};

VECTOR_IMPL(List, Value, true) 
	static bool isNA(Value const& c) { return c.isNil(); }
	static bool isCheckedNA(Value const& c) { return isNA(c); }
	static bool isNaN(Value const& c) { return false; }
	static bool isFinite(Value const& c) { return false; }
	static bool isInfinite(Value const& c) { return false; }
};


class Dictionary : public HeapObject {
protected:
	uint64_t size, load, ksize;
	
	struct Inner : public HeapObject {
		Pair d[];
	};

	Inner* d;

	// Returns the location of variable `name` in this environment or
	// an empty pair (String::NA, Value::Nil).
	// success is set to true if the variable is found. This boolean flag
	// is necessary for compiler optimizations to eliminate expensive control flow.
	Pair* find(String name, bool& success) const ALWAYS_INLINE {
		uint64_t i = ((uint64_t)name >> 3) & ksize;
		Pair* first = &d->d[i];
		if(__builtin_expect(first->n == name, true)) {
			success = true;
			return first;
		}
		uint64_t j = 0;
		while(d->d[i].n != Strings::NA) {
			i = (i+(++j)) & ksize;
			if(__builtin_expect(d->d[i].n == name, true)) {
				success = true;
				return &d->d[i];
			}
		}
		success = false;
		return &d->d[i];
	}

	// Returns the location where variable `name` should be inserted.
	// Assumes that `name` doesn't exist in the hash table yet.
	// Used for rehash and insert where this is known to be true.
	Pair* slot(String name) const ALWAYS_INLINE {
		uint64_t i = ((uint64_t)name >> 3) & ksize;
		if(__builtin_expect(d->d[i].n == Strings::NA, true)) {
			return &d->d[i];
		}
		uint64_t j = 0;
		while(d->d[i].n != Strings::NA) {
			i = (i+(++j)) & ksize;
		}
		return &d->d[i];
	}

	void rehash(uint64_t s) {
		uint64_t old_size = size;
		uint64_t old_load = load;
		Inner* old_d = d;

		d = new (sizeof(Pair)*s) Inner();
		size = s;
		ksize = s-1;
		clear();
		
		// copy over previous populated values...
		if(old_load > 0) {
			for(uint64_t i = 0; i < old_size; i++) {
				if(old_d->d[i].n != Strings::NA) {
					load++;
					*slot(old_d->d[i].n) = old_d->d[i];
				}
			}
		}
	}

public:
	Dictionary(int64_t initialLoad) : size(0), load(0), d(0) {
		rehash(std::max((uint64_t)1, nextPow2(initialLoad*2)));
	}

	bool has(String name) const ALWAYS_INLINE {
		bool success;
		find(name, success);
		return success;
	}

	Value const& get(String name) const ALWAYS_INLINE {
		bool success;
		return find(name, success)->v;
	}

	Value& insert(String name) ALWAYS_INLINE {
		bool success;
		Pair* p = find(name, success);
		if(!success) {
			if(((load+1) * 2) > size)
				rehash((size*2));
			load++;
			p = slot(name);
			p->n = name;
		}
		return p->v;
	}

	void remove(String name) {
		bool success;
		Pair* p = find(name, success);
		if(success) {
			load--;
			memset(p, 0, sizeof(Pair));
		}
	}

	void clear() {
		memset(d->d, 0, sizeof(Pair)*size); 
		load = 0;
	}

	// clone with room for extra elements
	Dictionary* clone(uint64_t extra) const {
		Dictionary* clone = new Dictionary((load+extra)*2);
		// copy over elements
		if(load > 0) {
			for(uint64_t i = 0; i < size; i++) {
				if(d->d[i].n != Strings::NA) {
					clone->load++;
					*clone->slot(d->d[i].n) = d->d[i];
				}
			}
		}
		return clone;
	}

	class const_iterator {
		Dictionary const* d;
		int64_t i;
	public:
		const_iterator(Dictionary const* d, int64_t idx) {
			this->d = d;
			i = std::max((int64_t)0, std::min((int64_t)d->size, idx));
			while(d->d->d[i].n == Strings::NA && i < (int64_t)d->size) i++;
		}
		String string() const { return d->d->d[i].n; }	
		Value const& value() const { return d->d->d[i].v; }
		const_iterator& operator++() {
			while(d->d->d[++i].n == Strings::NA && i < (int64_t)d->size);
			return *this;
		}
		bool operator==(const_iterator const& o) {
			return d == o.d && i == o.i;
		}
		bool operator!=(const_iterator const& o) {
			return d != o.d || i != o.i;
		}
	};

	const_iterator begin() const {
		return const_iterator(this, 0);
	}

	const_iterator end() const {
		return const_iterator(this, size);
	}

	 void visit() const;
};

class Environment : public Dictionary {
public:
	Environment* lexical, *dynamic;
	Value call;
	PairList dots;
	bool named;	// true if any of the dots have names	

	explicit Environment(int64_t initialLoad, Environment* lexical, Environment* dynamic, Value const& call) :
			Dictionary(initialLoad), 
			lexical(lexical), dynamic(dynamic), call(call), named(false) {}

	Environment* LexicalScope() const { return lexical; }
	Environment* DynamicScope() const { return dynamic; }

	// Look up insertion location using R <<- rules
	// (i.e. find variable with same name in the lexical scope)
	Value& insertRecursive(String name, Environment*& env) const ALWAYS_INLINE {
		bool success;
		env = (Environment*)this;
		Pair* p = env->find(name, success);
		while(!success && (env = env->LexicalScope())) {
			p = env->find(name, success);
		}
		return p->v;
	}
	
	// Look up variable using standard R lexical scoping rules
	// Should be same as insertRecursive, but with extra constness
	Value const& getRecursive(String name, Environment*& env) const ALWAYS_INLINE {
		return insertRecursive(name, env);
	}

	struct Pointer {
		Environment* env;
		String name;
	};

	Pointer makePointer(String name) {
		return (Pointer) { this, name };
	}

	static Value const& getPointer(Pointer const& p) {
		return p.env->get(p.name);
	}

	static void assignPointer(Pointer const& p, Value const& value) {
		p.env->insert(p.name) = value;
	}
	
	void visit() const;
};

#endif
