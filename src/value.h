
#ifndef _RIPOSTE_VALUE_H
#define _RIPOSTE_VALUE_H

#include <vector>
#include <assert.h>
#include <limits>

#include "common.h"
#include "type.h"
#include "bc.h"
#include "rgc.h"
#include "strings.h"
#include "exceptions.h"

typedef int64_t IRef;

struct Value {
	
	union {
		struct {
			Type::Enum type:4;
			int64_t length:60;
		};
		int64_t header;
	};
	union {
		void* p;
		int64_t i;
		double d;
		char c;
		String s;
		struct {
			Type::Enum typ;
			uint16_t ref;
		} future;
	};

	static void Init(Value& v, Type::Enum type, int64_t length) {
		v.header =  type + (length<<4);
	}

	// Warning: shallow equality!
	bool operator==(Value const& other) const {
		return header == other.header && p == other.p;
	}
	
	bool operator!=(Value const& other) const {
		return header != other.header || p != other.p;
	}

	// Ordering solely for the purpose of putting them in a map
	bool operator<(Value const& other) const {
		return header < other.header || (header == other.header && p < other.p);
	}
	
	bool isNil() const { return header == 0; }
	bool isNull() const { return type == Type::Null; }
	bool isLogical() const { return type == Type::Logical; }
	bool isInteger() const { return type == Type::Integer; }
	bool isLogical1() const { return header == (1<<4) + Type::Logical; }
	bool isInteger1() const { return header == (1<<4) + Type::Integer; }
	bool isDouble() const { return type == Type::Double; }
	bool isDouble1() const { return header == (1<<4) + Type::Double; }
	bool isCharacter() const { return type == Type::Character; }
	bool isCharacter1() const { return header == (1<<4) + Type::Character; }
	bool isList() const { return type == Type::List; }
	bool isPromise() const { return type == Type::Promise && !isNil(); }
	bool isDefault() const { return type == Type::Default; }
	bool isDotdot() const { return type == Type::Dotdot; }
	bool isFuture() const { return type == Type::Future; }
	bool isFunction() const { return type == Type::Function; }
	bool isObject() const { return type == Type::Object; }
	bool isMathCoerce() const { return isDouble() || isInteger() || isLogical(); }
	bool isLogicalCoerce() const { return isDouble() || isInteger() || isLogical(); }
	bool isVector() const { return isNull() || isLogical() || isInteger() || isDouble() || isCharacter() || isList(); }
	bool isClosureSafe() const { return isNull() || isLogical() || isInteger() || isDouble() || isFuture() || isCharacter() || (isList() && length==0); }
	bool isConcrete() const { return type > Type::Dotdot; }

	bool isScalar() const { return length == 1; }

	template<class T> T& scalar() { throw "not allowed"; }
	template<class T> T const& scalar() const { throw "not allowed"; }

	static Value const& Nil() { static const Value v = { {{Type::Promise, 0}}, {0} }; return v; }

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

template<Type::Enum VType, typename ElementType, bool Recursive>
struct Vector : public Value {

	typedef ElementType Element;
	static const Type::Enum ValueType = VType;
	static const bool canPack = sizeof(ElementType) <= sizeof(int64_t) && !Recursive;

	struct Inner {
		int64_t length;
		int64_t padding;
		ElementType data[];
		//virtual void visit() const;
		//Inner(int64_t length) : length(length) {}
	};

	ElementType const* v() const { 
		return (canPack && isScalar()) ? 
			&Value::scalar<ElementType>() : ((Inner*)p)->data; 
	}
	ElementType* v() { 
		return (canPack && isScalar()) ? 
			&Value::scalar<ElementType>() : ((Inner*)p)->data; 
	}

	Inner* inner() const {
		return (length > canPack) ? (Inner*)p : 0;
	}
	
	ElementType& operator[](int64_t index) { return v()[index]; }
	ElementType const& operator[](int64_t index) const { return v()[index]; }

	static Vector<VType, ElementType, Recursive>& Init(Value& v, int64_t length) {
		Value::Init(v, ValueType, length);
		if((canPack && length > 1) || (!canPack && length > 0)) {
			int64_t l = length;
			// round l up to nearest even number so SSE can work on tail region
			l += (int64_t)((uint64_t)l & 1);
			int64_t length_aligned = (l < 128) ? (l + 1) : l;
			//v.p = new (sizeof(Element)*length_aligned) Inner(length);
			v.p = malloc(sizeof(Element)*length_aligned + sizeof(Inner));
			assert(l < 128 || (0xF & (int64_t)v.p) == 0);
			if( (0xF & (int64_t)v.p) != 0)
				v.p =  (char*)v.p + 0x8;
		}
		return (Vector<VType, ElementType, Recursive>&)v;
	}

	static void InitScalar(Value& v, ElementType const& d) {
		Value::Init(v, ValueType, 1);
		if(canPack)
			v.scalar<ElementType>() = d;
		else {
			//Inner* i = new (sizeof(Element)*4) Inner(1);
			//i->data[0] = d;
			//v.p = i;
			Inner* i = (Inner*)malloc(sizeof(Element)*4 + sizeof(Inner));
			i->data[0] = d;
			v.p = i;
			//i->length = 1;
			//v.p = malloc(sizeof(Element)*4) :
			//		malloc(sizeof(Element)*4);
			//*(Element*)v.p = d;
		}
	}
};

union _doublena {
	int64_t i;
	double d;
};


#define VECTOR_IMPL(Name, Element, Recursive) 				\
struct Name : public Vector<Type::Name, Element, Recursive> { 			\
	explicit Name(int64_t length=0) { Init(*this, length); } 	\
	static Name c() { Name c(0); return c; } \
	static Name c(Element v0) { Name c(1); c[0] = v0; return c; } \
	static Name c(Element v0, Element v1) { Name c(2); c[0] = v0; c[1] = v1; return c; } \
	static Name c(Element v0, Element v1, Element v2) { Name c(3); c[0] = v0; c[1] = v1; c[2] = v2; return c; } \
	static Name c(Element v0, Element v1, Element v2, Element v3) { Name c(4); c[0] = v0; c[1] = v1; c[2] = v2; c[3] = v3; return c; } \
	const static Element NAelement; \
	static Name NA() { static Name na = Name::c(NAelement); return na; }  \
	static Name& Init(Value& v, int64_t length) { return (Name&)Vector<Type::Name, Element, Recursive>::Init(v, length); } \
	static void InitScalar(Value& v, Element const& d) { Vector<Type::Name, Element, Recursive>::InitScalar(v, d); }\
/* note missing }; */

VECTOR_IMPL(Null, unsigned char, false)  
	static Null Singleton() { static Null s = Null::c(); return s; } 
	static bool isNA() { return false; }
	static bool isCheckedNA() { return false; }
};

VECTOR_IMPL(Logical, char, false)
	const static char TrueElement;
	const static char FalseElement;

	static Logical True() { static Logical t = Logical::c(-1); return t; }
	static Logical False() { static Logical f = Logical::c(0); return f; } 
	
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
	static Double Inf() { static Double i = Double::c(std::numeric_limits<double>::infinity()); return i; }
	static Double NInf() { static Double i = Double::c(-std::numeric_limits<double>::infinity()); return i; }
	static Double NaN() { static Double n = Double::c(std::numeric_limits<double>::quiet_NaN()); return n; } 
	
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

/*template<Type::Enum VType, typename ElementType, bool Recursive>
void Vector<VType, ElementType, Recursive>::Inner::visit() const {}

template<>
void Vector<Type::List, Value, true>::Inner::visit() const;
*/
struct Future : public Value {
	static const Type::Enum ValueType = Type::Future;
	static Future& Init(Value& f, Type::Enum typ,int64_t length,IRef ref) {
		Value::Init(f,Type::Future,length);
		f.future.ref = ref;
		f.future.typ = typ;
		return (Future&)f;
	}
};

struct Function : public Value {
	static const Type::Enum ValueType = Type::Function;
	static Function& Init(Value& v, Prototype* proto, Environment* env) {
		v.header = (int64_t)proto + Type::Function;
		v.p = env;
		return (Function&)v;
	}

	Prototype* prototype() const { return (Prototype*)(length << 4); }
	Environment* environment() const { return (Environment*)p; }
};

struct Promise : public Value {
	static const Type::Enum ValueType = Type::Promise;
	static Promise& Init(Value& v, Prototype* proto, Environment* env) {
		v.header = (int64_t)proto + Type::Promise;
		v.p = env;
		return (Promise&)v;
	}

	Prototype* prototype() const { return (Prototype*)(length << 4); }
	Environment* environment() const { return (Environment*)p; }
};

struct Default : public Value {
	static const Type::Enum ValueType = Type::Default;
	static Default& Init(Value& v, Prototype* proto, Environment* env) {
		v.header = (int64_t)proto + Type::Default;
		v.p = env;
		return (Default&)v;
	}

	Prototype* prototype() const { return (Prototype*)(length << 4); }
	Environment* environment() const { return (Environment*)p; }
};

class Dictionary : public HeapObject {
protected:
	static const uint64_t inlineSize = 8;
	uint64_t size, load;
	
	struct Inner : public HeapObject {
		Pair d[];
		virtual void visit() const;
	};

	Inner* d;

	uint64_t hash(String s) const ALWAYS_INLINE { return (uint64_t)s>>3; }

	// Returns the location of variable `name` in this environment or
	// an empty pair (String::NA, Value::Nil).
	// success is set to true if the variable is found. This boolean flag
	// is necessary for compiler optimizations to eliminate expensive control flow.
	Pair* find(String name, bool& success) const ALWAYS_INLINE {
		uint64_t i = hash(name) & (size-1);
		if(__builtin_expect(d->d[i].n == name, true)) {
			success = true;
			return &d->d[i];
		}
		uint64_t j = 0;
		while(d->d[i].n != Strings::NA) {
			i = (i+(++j)) & (size-1);
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
		uint64_t i = hash(name) & (size-1);
		if(__builtin_expect(d->d[i].n == Strings::NA, true)) {
			return &d->d[i];
		}
		uint64_t j = 0;
		while(d->d[i].n != Strings::NA) {
			i = (i+(++j)) & (size-1);
		}
		return &d->d[i];
	}

	void rehash(uint64_t s) {
		uint64_t old_size = size;
		uint64_t old_load = load;
		Inner* old_d = d;

		s = nextPow2(s);
		if(s <= size) return; // should rehash on shrinking sometimes, when?

		size = s;
		d = new (sizeof(Pair)*s) Inner();
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
	Dictionary() : size(inlineSize), d(new (sizeof(Pair) * 8) Inner()) {
		clear();
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
		load = 0;
		memset(d->d, 0, sizeof(Pair)*size); 
	}

	// clone with room for extra elements
	Dictionary* clone(uint64_t extra) const {
		Dictionary* clone = new Dictionary();
		clone->rehash(size+extra);
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

	virtual void visit() const;
};

// Object implements an immutable dictionary interface.
// Objects also have a base value which right now must be a non-object type...
//  However S4 objects can contain S3 objects so we may have to change this.
//  If we make this change, then all code that unwraps objects must do so recursively.
class Object : public Value {
	
private:
	struct Inner {
		Value base;
		Dictionary* d;
		Inner(Value const& base, Dictionary* d) : base(base), d(d) {}
	};

public:
	static const Type::Enum ValueType = Type::Object;
	
	Object() {}
	
	static void Init(Object& o, Value const& base, Dictionary* dictionary=0) {
		// Create inner first works if base and o overlap.
		Inner* p = new Inner(base, dictionary == 0 ? new Dictionary() : dictionary);
		Value::Init(o, Type::Object, 0);
		o.p = p;
	}

	Value const& base() const {
		return ((Inner const*)p)->base;
	}

	Dictionary* dictionary() const {
		return ((Inner const*)p)->d;
	}

	bool has(String name) const {
		return ((Inner const*)p)->d->has(name);
	}
	
	Value const& get(String name) const {
		return ((Inner const*)p)->d->get(name);
	}

	void insertMutable(String name, Value const& v) {
		if(!v.isNil())
			((Inner*)p)->d->insert(name) = v;
	}

	Object insert(String name, Value const& v) {
		Object o;
		Object::Init(o, ((Inner*)p)->base, ((Inner*)p)->d->clone(1));
		o.insertMutable(name, v);
		return o;
	}
};

class Environment : public Dictionary {
private:
	Environment* lexical, *dynamic;
	
public:
	Value call;
	PairList dots;
	bool named;	// true if any of the dots have names	

	explicit Environment(Environment* lexical=0, Environment* dynamic=0) : 
			lexical(lexical), dynamic(dynamic), call(Null::Singleton()) {}

	void init(Environment* l, Environment* d, Value const& call) {
		lexical = l;
		dynamic = d;
		this->call = call;
		clear();
		dots.clear();
	}
	
	Environment* LexicalScope() const { return lexical; }
	Environment* DynamicScope() const { return dynamic; }

	// Look up insertion location using R <<- rules
	// (i.e. find variable with same name in the lexical scope)
	Value& insertRecursive(String name) const ALWAYS_INLINE {
		bool success;
		Environment const* env = this;
		Pair* p = env->find(name, success);
		while(!success && env->LexicalScope()) {
			env = env->LexicalScope();
			p = env->find(name, success);
		}
		return p->v;
	}
	
	// Look up variable using standard R lexical scoping rules
	// Should be same as insertRecursive, but with extra constness
	Value const& getRecursive(String name) const ALWAYS_INLINE {
		return insertRecursive(name);
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
	
	virtual void visit() const;
};

struct REnvironment : public Value {
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

#endif
