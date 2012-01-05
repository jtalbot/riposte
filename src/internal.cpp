
#include "internal.h"
#include "compiler.h"
#include "parser.h"
#include "library.h"
#include <math.h>
#include <fstream>

#include <pthread.h>

template<typename T>
T const& Cast(Value const& v) {
	if(v.type != T::VectorType) _error("incorrect type passed to internal function");
	return (T const&)v;
}

void cat(Thread& thread, Value const* args, Value& result) {
	List const& a = Cast<List>(args[0]);
	for(int64_t i = 0; i < a.length; i++) {
		if(!List::isNA(a[i])) {
			Character c = As<Character>(thread, a[i]);
			for(int64_t j = 0; j < c.length; j++) {
				printf("%s", thread.externStr(c[j]).c_str());
				if(j < c.length-1) printf(" ");
			}
		}
	}
	result = Null::Singleton();
}

void remove(Thread& thread, Value const* args, Value& result) {
	Character const& a = Cast<Character>(args[0]);
	REnvironment e(args[1]);
	for(int64_t i = 0; i < a.length; i++) {
		e.ptr()->remove(a[i]);
	}
	result = Null::Singleton();
}

void library(Thread& thread, Value const* args, Value& result) {
	Character from = As<Character>(thread, args[0]);
	if(from.length > 0) {
		loadLibrary(thread, "library", thread.externStr(from[0]));
	}
	result = Null::Singleton();
}

void sequence(Thread& thread, Value const* args, Value& result) {
	double from = asReal1(args[0]);
	double by = asReal1(args[1]);
	double len = asReal1(args[2]);

	result = Sequence(from, by, len);	
}

void repeat(Thread& thread, Value const* args, Value& result) {
	double v = asReal1(args[0]);
	//double e = asReal1(args[1]);
	double l = asReal1(args[2]);
	
	Double r(l);
	for(int64_t i = 0; i < l; i++) {
		r[i] = v;
	}
	result = r;
}

void inherits(Thread& thread, Value const* args, Value& result) {
	Value x = args[0];
	Character what = Cast<Character>(args[1]);
	Logical which = Cast<Logical>(args[2]);
	// NYI: which
	Character c = klass(thread, x);
	bool inherits = false;
	for(int64_t i = 0; i < what.length && !inherits; i++) {
		for(int64_t j = 0; j < c.length && !inherits; j++) {
			if(what[i] == c[j]) inherits = true;
		}
	}
	result = Logical::c(inherits);
}

void attr(Thread& thread, Value const* args, Value& result)
{
	// NYI: exact
	Value object = args[0];
	if(object.isObject()) {
		Character which = Cast<Character>(args[1]);
		result = ((Object const&)object).getAttribute(which[0]);
	}
	else {
		result = Value::Nil();
	}
}

void assignAttr(Thread& thread, Value const* args, Value& result)
{
	Value object = args[0];
	Character which = Cast<Character>(args[1]);
	if(!object.isObject()) {
		Value v;
		Object::Init(v, object);
		object = v;
	}
	result = ((Object&)object).setAttribute(which[0], args[2]);
}

Type::Enum cTypeCast(Type::Enum s, Type::Enum t)
{
	if(s == Type::Object || t == Type::Object) return Type::List;
	else return std::max(s, t);
}

// These are all tree-based reductions. Should we have a tree reduction byte code?
int64_t unlistLength(Thread& thread, int64_t recurse, Value a) {
	if(a.isObject()) a = ((Object&)a).base();
	if(recurse > 0 && a.isList()) {
		List l(a);
		int64_t t = 0;
		for(int64_t i = 0; i < l.length; i++) 
			t += unlistLength(thread, recurse-1, l[i]);
		return t;
	}
	else if(a.isVector()) return a.length;
	else return 1;
}

Type::Enum unlistType(Thread& thread, int64_t recurse, Value a) {
	if(a.isObject()) a = ((Object&)a).base();
	if(a.isList()) {
		List l(a);
		Type::Enum t = Type::Null;
		for(int64_t i = 0; i < l.length; i++) 
			t = cTypeCast(recurse > 0 ? unlistType(thread, recurse-1, l[i]) : l[i].type, t);
		return t;
	}
	else if(a.isVector()) return a.type;
	else return Type::List;
}

template< class T >
void unlist(Thread& thread, int64_t recurse, Value a, T& out, int64_t& start) {
	if(a.isObject()) a = ((Object&)a).base();
	if(recurse > 0 && a.isList()) {
		List l(a);
		for(int64_t i = 0; i < l.length; i++) 
			unlist(thread, recurse-1, l[i], out, start);
		return;
	}
	else if(a.isVector()) { Insert(thread, a, 0, out, start, a.length); start += a.length; }
	else _error("Unexpected non-basic type in unlist");
}

template<>
void unlist<List>(Thread& thread, int64_t recurse, Value a, List& out, int64_t& start) {
	if(a.isObject()) a = ((Object&)a).base();
	if(recurse > 0 && a.isList()) {
		List l(a);
		for(int64_t i = 0; i < l.length; i++) 
			unlist(thread, recurse-1, l[i], out, start);
		return;
	}
	else if(a.isVector()) { Insert(thread, a, 0, out, start, a.length); start += a.length; }
	else out[start++] = a;
}

bool unlistHasNames(Thread& thread, int64_t recurse, Value a) {
	if(a.isObject() && ((Object const&)a).hasNames()) return true;
	if(a.isObject()) a = ((Object&)a).base();
	if(recurse > 0 && a.isList()) {
		List l(a);
		bool hasNames = false;
		for(int64_t i = 0; i < l.length; i++) 
			hasNames = hasNames || unlistHasNames(thread, recurse-1, l[i]);
		return hasNames;
	}
	else return false;
}

std::string makeName(Thread& thread, std::string prefix, String name, int64_t i) {
	if(prefix.length() > 0) {
		if(name != Strings::empty)
			return prefix + "." + thread.externStr(name);
		else
			return prefix + intToStr(i+1);
	}
	else {
		return thread.externStr(name);
	}
}

void unlistNames(Thread& thread, int64_t recurse, Value a, Character& out, int64_t& start, std::string prefix) {
	Character names(0);
	if(a.isObject() && ((Object&)a).hasNames()) {
		names = (Character)((Object&)a).getNames();
	}
	if(a.isObject()) a = ((Object&)a).base();

	if(recurse > 0 && a.isList()) {
		List l(a);
		for(int64_t i = 0; i < l.length; i++)
			unlistNames(thread, recurse-1, l[i], out, start, makeName(thread, prefix, i < names.length ? names[i] : Strings::empty, i));
		return;
	}
	else if(a.isVector() && a.length != 1) { 
		for(int64_t i = 0; i < a.length; i++) {
			out[start++] = thread.internStr(makeName(thread, prefix, (i < names.length) ? names[i] : Strings::empty, i));
		}
	}
	else out[start++] = thread.internStr(prefix);
}

// TODO: useNames parameter could be handled at the R level
void unlist(Thread& thread, Value const* args, Value& result) {
	Value v = args[0];
	int64_t recurse = Cast<Logical>(args[1])[0] ? std::numeric_limits<int64_t>::max() : 1;
	bool useNames = Cast<Logical>(args[2])[0];
	
	int64_t length = unlistLength(thread, recurse, v);
	Type::Enum type = unlistType(thread, recurse, v);

	switch(type) {
		#define CASE(Name) \
			case Type::Name: { \
				Name out(length); \
				int64_t i = 0; \
				unlist(thread, recurse, v, out, i); \
				result = out; \
			} break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Insert into this type"); break;
	};

	if(useNames && unlistHasNames(thread, recurse, v)) {
		Character outnames(length);
		int64_t i = 0;
		unlistNames(thread, recurse, v, outnames, i, "");
		Object::Init(result, result, outnames);
	}
}


template< class A >
struct SubsetInclude {
	static void eval(Thread& thread, A const& a, Integer const& d, int64_t nonzero, Value& out)
	{
		A r(nonzero);
		int64_t j = 0;
		typename A::Element const* ae = a.v();
		typename Integer::Element const* de = d.v();
		typename A::Element* re = r.v();
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) {
			if(Integer::isNA(de[i])) re[j++] = A::NAelement;	
			else if(de[i] != 0) re[j++] = ae[de[i]-1];
		}
		out = r;
	}
};

template< class A >
struct SubsetExclude {
	static void eval(Thread& thread, A const& a, Integer const& d, int64_t nonzero, Value& out)
	{
		std::set<Integer::Element> index; 
		typename A::Element const* ae = a.v();
		typename Integer::Element const* de = d.v();
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) if(-de[i] > 0 && -de[i] <= (int64_t)a.length) index.insert(-de[i]);
		// iterate through excluded elements copying intervening ranges.
		A r(a.length-index.size());
		typename A::Element* re = r.v();
		int64_t start = 1;
		int64_t k = 0;
		for(std::set<Integer::Element>::const_iterator i = index.begin(); i != index.end(); ++i) {
			int64_t end = *i;
			for(int64_t j = start; j < end; j++) re[k++] = ae[j-1];
			start = end+1;
		}
		for(int64_t j = start; j <= a.length; j++) re[k++] = ae[j-1];
		out = r;
	}
};

template< class A >
struct SubsetLogical {
	static void eval(Thread& thread, A const& a, Logical const& d, Value& out)
	{
		typename A::Element const* ae = a.v();
		typename Logical::Element const* de = d.v();
		// determine length
		int64_t length = 0;
		if(d.length > 0) {
			int64_t j = 0;
			int64_t maxlength = std::max(a.length, d.length);
			for(int64_t i = 0; i < maxlength; i++) {
				if(!Logical::isFalse(de[j])) length++;
				if(++j >= d.length) j = 0;
			}
		}
		A r(length);
		typename A::Element* re = r.v();
		int64_t j = 0, k = 0;
		for(int64_t i = 0; i < std::max(a.length, d.length) && k < length; i++) {
			if(i >= a.length || Logical::isNA(de[j])) re[k++] = A::NAelement;
			else if(Logical::isTrue(de[j])) re[k++] = ae[i];
			if(++j >= d.length) j = 0;
		}
		out = r;
	}
};

template< class A >
inline int64_t find(Thread& thread, A const& a, typename A::Element const& b) {
	typename A::Element const* ae = a.v();
	// eventually use an index here...
	for(int64_t i = 0; i < a.length; i++) {
		if(ae[i] == b) return i;
	}
	return -1;
}

template< class A, class B >
struct SubsetIndexed {
	static void eval(Thread& thread, A const& a, B const& b, B const& d, Value& out)
	{
		typename A::Element const* ae = a.v();
		typename B::Element const* de = d.v();
		
		int64_t length = d.length;
		A r(length);
		for(int64_t i = 0; i < length; i++) {
			int64_t index = find(thread, b, de[i]);
			r[i] = index >= 0 ? ae[index] : A::NAelement;
		}
		out = r;
	}
};

void SubsetSlow(Thread& thread, Value const& a, Value const& i, Value& out) {
	if(i.isDouble() || i.isInteger()) {
		if(a.isObject()) {
			// TODO: this computes positive twice
			Value r, names;
			SubsetSlow(thread, ((Object const&)a).base(), i, r);
			if(((Object const&)a).hasNames()) {
				SubsetSlow(thread, ((Object const&)a).getNames(), i, names);
				Object::Init(r, r, names);
			}
			out = r;
			return;
		}
		Integer index = As<Integer>(thread, i);
		int64_t positive = 0, negative = 0;
		for(int64_t i = 0; i < index.length; i++) {
			if(index[i] > 0 || Integer::isNA(index[i])) positive++;
			else if(index[i] < 0) negative++;
		}
		if(positive > 0 && negative > 0)
			_error("mixed subscripts not allowed");
		else if(positive > 0) {
			switch(a.type) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name,...) case Type::Name: SubsetInclude<Name>::eval(thread, (Name const&)a, index, positive, out); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};
		}
		else if(negative > 0) {
			switch(a.type) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetExclude<Name>::eval(thread, (Name const&)a, index, negative, out); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};	
		}
		else {
			switch(a.type) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: out = Name(0); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};	
		}
	}
	else if(i.isLogical()) {
		if(a.isObject()) {
			Value r, names;
			SubsetSlow(thread, ((Object const&)a).base(), i, r);
			if(((Object const&)a).hasNames()) {
				SubsetSlow(thread, ((Object const&)a).getNames(), i, names);
				Object::Init(r, r, names);
			}
			out = r;
			return;
		}
		Logical index = Logical(i);
		switch(a.type) {
			case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetLogical<Name>::eval(thread, (Name const&)a, index, out); break;
					 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
		};	
	}
	else if(i.isCharacter()) {
		if(a.isObject() && ((Object const&)a).hasNames()) {
			Value const& b = ((Object const&)a).base();
			Value const& n = ((Object const&)a).getNames();
			switch(b.type) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetIndexed<Name, Character>::eval(thread, (Name const&)b, (Character const&)n, (Character const&)i, out); break;
				VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};
			Object::Init(out, out, (Character const&)i);	
		}
		else _error("Object does not have names for subsetting");
	}
	else {
		_error("NYI indexing type");
	}
}

template< class A  >
struct SubsetAssignInclude {
	static void eval(Thread& thread, A const& a, bool clone, Integer const& d, A const& b, Value& out)
	{
		typename A::Element const* be = b.v();
		typename Integer::Element const* de = d.v();

		// compute max index 
		int64_t outlength = a.length;
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) {
			outlength = std::max(outlength, de[i]);
		}

		// should use max index here to extend vector if necessary	
		A r = a;
		Resize(thread, clone, r, outlength);
		typename A::Element* re = r.v();
		for(int64_t i = 0, j = 0; i < length; i++, j++) {	
			if(j >= b.length) j = 0;
			int64_t idx = de[i];
			if(idx != 0)
				re[idx-1] = be[j];
		}
		out = r;
	}
};

template< class A >
struct SubsetAssignLogical {
	static void eval(Thread& thread, A const& a, bool clone, Logical const& d, A const& b, Value& out)
	{
		typename A::Element const* be = b.v();
		typename Logical::Element const* de = d.v();
		
		// determine length
		int64_t length = std::max(a.length, d.length);
		A r = a;
		Resize(thread, clone, r, length);
		typename A::Element* re = r.v();
		int64_t j = 0, k = 0;
		for(int64_t i = 0; i < length; i++) {
			if(i >= a.length && !Logical::isTrue(de[j])) re[i] = A::NAelement;
			else if(Logical::isTrue(de[j])) re[i] = be[k++];
			if(++j >= d.length) j = 0;
			if(k >= b.length) k = 0;
		}
		out = r;
	}
};

template< class A, class B >
struct SubsetAssignIndexed {
	static void eval(Thread& thread, A const& a, bool clone, B const& b, B const& d, A const& v, Value& out)
	{
		typename B::Element const* de = d.v();
		
		std::map<typename B::Element, int64_t> overflow;
		int64_t extra = a.length;

		int64_t length = d.length;
		Integer include(length);
		for(int64_t i = 0; i < length; i++) {
			int64_t index = find(thread, b, de[i]);
			if(index < 0) {
				if(overflow.find(de[i]) != overflow.end())
					index = overflow[de[i]];
				else {
					index = extra++;
					overflow[de[i]] = index;
				}
			}
			include[i] = index+1;
		}
		SubsetAssignInclude<A>::eval(thread, a, clone, include, v, out);
	}
};

void SubsetAssignSlow(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(i.isDouble() || i.isInteger()) {
		Integer idx = As<Integer>(thread, i);
		switch(a.type) {
#define CASE(Name) case Type::Name: SubsetAssignInclude<Name>::eval(thread, (Name const&)a, clone, idx, As<Name>(thread, b), c); break;
			VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			case Type::Object: {
				Value r;
				SubsetAssignSlow(thread, ((Object const&)a).base(), clone, i, b, r);
				if(((Object const&)a).hasNames()) {
					Character names = (Character)((Object const&)a).getNames();
					Resize(thread, clone, names, r.length, Strings::empty);
					Object::Init(r, r, names);
				}
				c = r;
			} break;
			default: _error("NYI: subset assign type"); break;
		};
	}
	else if(i.isLogical()) {
		Logical index = Logical(i);
		switch(a.type) {
#define CASE(Name) case Type::Name: SubsetAssignLogical<Name>::eval(thread, (Name const&)a, clone, index, As<Name>(thread, b), c); break;
					 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			case Type::Object: {
				Value r;
				SubsetAssignSlow(thread, ((Object const&)a).base(), clone, i, b, r);
				if(((Object const&)a).hasNames()) {
					Character names = (Character)((Object const&)a).getNames();
					Resize(thread, clone, names, r.length, Strings::empty);
					Object::Init(r, r, names);
				}
				c = r;
			} break;
			default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
		};	
	}
	else if(i.isCharacter()) {
		if(a.isObject() && ((Object const&)a).hasNames()) {
			Value const& base = ((Object const&)a).base();
			Value const& names = ((Object const&)a).getNames();
			switch(base.type) {
				case Type::Null: c = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetAssignIndexed<Name, Character>::eval(thread, (Name const&)base, clone, (Character const&)names, (Character const&)i, As<Name>(thread, b), c); break;
				VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};
			Value newNames;
			SubsetAssignIndexed<Character, Character>::eval(thread, (Character const&)names, clone, (Character const&)names, (Character const&)i, (Character const&)i, newNames);
			Object::Init(c, c, (Character const&)newNames);
		}
		else _error("Object does not have names for subsetting");
	}
	else {
		_error("NYI indexing type");
	}
}

void Subset2AssignSlow(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(i.isDouble() || i.isInteger() || i.isCharacter())
		SubsetAssignSlow(thread, a, clone, i, b, c);
	// Frankly it seems pointless to support this case. We should make this an error.
	//else if(i.isLogical() && i.length == 1 && ((Logical const&)i)[0])
	//	SubsetAssignSlow(thread, a, clone, As<Integer>(thread, i), b, c);
	else
		_error("NYI indexing type");
}

void length(Thread& thread, Value const* args, Value& result) {
	result = Integer::c(args[0].length);
}

void eval_fn(Thread& thread, Value const* args, Value& result) {
	result = thread.eval(Compiler::compile(thread.state, args[0]), REnvironment(args[1]).ptr());
}

struct lapplyargs {
	List& in;
	List& out;
	Value func;
};

void lapplybody(void* args, uint64_t start, uint64_t end, Thread& thread) {
	lapplyargs& l = *(lapplyargs*)args;
	//printf("lapplybody called with %d to %d\n", start, end);
	List apply(2);
	apply[0] = l.func;
	apply[1] = Value::Nil();
	Prototype* p = Compiler::compile(thread.state, CreateCall(apply));
	for( size_t i=start; i!=end; ++i ) {
		p->calls[0].arguments[0] = l.in[i];
		l.out[i] = thread.eval(p);
	}
	//return 0;
}

void lapply(Thread& thread, Value const* args, Value& result) {
	List x = As<List>(thread, args[0]);
	Value func = args[1];
	List r(x.length);

	/*List apply(2);
	apply[0] = func;

	// TODO: should have a way to make a simple function call without compiling,
	// or should have a fast case for compilation
	Thread ithread(state);
	for(int64_t i = 0; i < x.length; i++) {
		apply[1] = x[i];
		r[i] = eval(ithread, Compiler::compile(thread, CreateCall(apply)));
	}*/

	/*List apply(2);
	apply[0] = func;
	apply[1] = Value::Nil();
	Prototype* p = Compiler::compile(thread, CreateCall(apply));
	Thread ithread(state);
	for(int64_t i = 0; i < x.length; i++) {
		p->calls[0].arguments[0] = x[i];
		r[i] = eval(ithread, p);
	}*/

	/*pthread_t h1, h2;

	lapplyargs a1 = (lapplyargs) {0, x.length/2, thread, x, r, func};
	lapplyargs a2 = (lapplyargs) {x.length/2, x.length, thread, x, r, func};

        pthread_create (&h1, NULL, lapplybody, &a1);
        pthread_create (&h2, NULL, lapplybody, &a2);
	pthread_join(h1, NULL);
	pthread_join(h2, NULL);
	*/

	lapplyargs a1 = (lapplyargs) {x, r, func};
	thread.doall(lapplybody, &a1, 0, x.length, 1, 1); 

	result = r;
}

/*
void tlist(Thread& thread, Value const* args, Value& result) {
	int64_t length = args.length > 0 ? 1 : 0;
	List a = Clone(args);
	for(int64_t i = 0; i < a.length; i++) {
		a[i] = force(thread, a[i]);
		if(a[i].isVector() && a[i].length != 0 && length != 0)
			length = std::max(length, (int64_t)a[i].length);
	}
	List r(length);
	for(int64_t i = 0; i < length; i++) {
		List element(args.length);
		for(int64_t j = 0; j < a.length; j++) {
			if(a[j].isVector())
				Element2(a[j], i%a[j].length, element[j]);
			else
				element[j] = a[j];
		}
		r[i] = element;
	}
	result = r;
}
*/
void source(Thread& thread, Value const* args, Value& result) {
	Character file = Cast<Character>(args[0]);
	std::ifstream t(thread.externStr(file[0]).c_str());
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string code = buffer.str();

	Parser parser(thread.state);
	Value value;
	parser.execute(code.c_str(), code.length(), true, value);	
	
	result = thread.eval(Compiler::compile(thread.state, value));
}

void environment(Thread& thread, Value const* args, Value& result) {
	Value e = args[0];
	if(e.isNull()) {
		result = REnvironment(thread.frame.environment);
		return;
	}
	else if(e.isFunction()) {
		result = REnvironment(Function(e).environment());
		return;
	}
	result = Null::Singleton();
}

// TODO: parent.frame and sys.call need to ignore frames for promises, etc. We may need
// the dynamic pointer in the environment after all...
void parentframe(Thread& thread, Value const* args, Value& result) {
	int64_t i = (int64_t)asReal1(args[0]);
	Environment* env = thread.frame.environment;
	while(i > 0 && env->DynamicScope() != 0) {
		env = env->DynamicScope();
		i--;
	}
	result = REnvironment(env);
}

void syscall(Thread& thread, Value const* args, Value& result) {
	int64_t i = (int64_t)asReal1(args[0]);
	Environment* env = thread.frame.environment;
	while(i > 0 && env->DynamicScope() != 0) {
		env = env->DynamicScope();
		i--;
	}
	result = env->call;
}

void stop_fn(Thread& thread, Value const* args, Value& result) {
	// this should stop whether or not the arguments are correct...
	std::string message = thread.externStr(Cast<Character>(args[0])[0]);
	_error(message);
	result = Null::Singleton();
}

void warning_fn(Thread& thread, Value const* args, Value& result) {
	std::string message = thread.externStr(Cast<Character>(args[0])[0]);
	_warning(thread, message);
	result = Character::c(thread.internStr(message));
} 
void isna_fn(Thread& thread, Value const* args, Value& result) {
	unaryFilter<Zip1, IsNAOp>(thread, args[0], result);
}

void isnan_fn(Thread& thread, Value const* args, Value& result) {
	unaryFilter<Zip1, IsNaNOp>(thread, args[0], result);
}

void nchar_fn(Thread& thread, Value const* args, Value& result) {
	// NYI: type or allowNA
	unaryCharacter<Zip1, NcharOp>(thread, args[0], result);
}

void nzchar_fn(Thread& thread, Value const* args, Value& result) {
	unaryCharacter<Zip1, NzcharOp>(thread, args[0], result);
}

void isfinite_fn(Thread& thread, Value const* args, Value& result) {
	unaryFilter<Zip1, IsFiniteOp>(thread, args[0], result);
}

void isinfinite_fn(Thread& thread, Value const* args, Value& result) {
	unaryFilter<Zip1, IsInfiniteOp>(thread, args[0], result);
}

void paste(Thread& thread, Value const* args, Value& result) {
	Character a = As<Character>(thread, args[0]);
	Character sep = As<Character>(thread, args[1]);
	std::string r = "";
	for(int64_t i = 0; i+1 < a.length; i++) {
		r = r + thread.externStr(a[i]) + thread.externStr(sep[0]);
	}
	if(a.length > 0) r = r + thread.externStr(a[a.length-1]);
	result = Character::c(thread.internStr(r));
}

void deparse(Thread& thread, Value const* args, Value& result) {
	result = Character::c(thread.internStr(thread.deparse(args[0])));
}

void substitute(Thread& thread, Value const* args, Value& result) {
	Value v = args[0];
	while(v.isPromise()) v = Function(v).prototype()->expression;
	
	if(isSymbol(v)) {
		Value r = thread.frame.environment->get(SymbolStr(v));
		if(!r.isNil()) v = r;
		while(v.isPromise()) v = Function(v).prototype()->expression;
	}
 	result = v;
}

void type_of(Thread& thread, Value const* args, Value& result) {
	// Should have a direct mapping from type to symbol.
	result = Character::c(thread.internStr(Type::toString(args[0].type)));
}

void exists(Thread& thread, Value const* args, Value& result) {
	Character c = As<Character>(thread, args[0]);
	REnvironment e(args[1]);
	Value v = e.ptr()->get(c[0]);
	if(v.isNil())
		result = Logical::False();
	else
		result = Logical::True();
}

#include <sys/time.h>

uint64_t readTime()
{
  timeval time_tt;
  gettimeofday(&time_tt, NULL);
  return (uint64_t)time_tt.tv_sec * 1000 * 1000 + (uint64_t)time_tt.tv_usec;
}

void proctime(Thread& thread, Value const* args, Value& result) {
	uint64_t s = readTime();
	result = Double::c(s/(1000000.0));
}

void traceconfig(Thread & thread, Value const* args, Value& result) {
	Integer c = As<Integer>(thread, args[0]);
	if(c.length == 0) _error("condition is of zero length");
	//thread.tracing.config = (TraceThread::Mode) c[0];
		
	std::vector<Thread*>& threads = thread.state.threads;
	for(uint64_t i = 0; i < threads.size(); i++) {
		threads[i]->tracing.config = (TraceThread::Mode) c[0];
	}

	result = Null::Singleton();
}

void registerCoreFunctions(State& state)
{
	state.registerInternalFunction(state.internStr("nchar"), (nchar_fn), 1);
	state.registerInternalFunction(state.internStr("nzchar"), (nzchar_fn), 1);
	state.registerInternalFunction(state.internStr("is.na"), (isna_fn), 1);
	state.registerInternalFunction(state.internStr("is.nan"), (isnan_fn), 1);
	state.registerInternalFunction(state.internStr("is.finite"), (isfinite_fn), 1);
	state.registerInternalFunction(state.internStr("is.infinite"), (isinfinite_fn), 1);
	
	state.registerInternalFunction(state.internStr("cat"), (cat), 1);
	state.registerInternalFunction(state.internStr("library"), (library), 1);
	state.registerInternalFunction(state.internStr("inherits"), (inherits), 3);
	
	state.registerInternalFunction(state.internStr("seq"), (sequence), 3);
	state.registerInternalFunction(state.internStr("rep"), (repeat), 3);
	
	state.registerInternalFunction(state.internStr("attr"), (attr), 3);
	state.registerInternalFunction(state.internStr("attr<-"), (assignAttr), 3);
	
	state.registerInternalFunction(state.internStr("unlist"), (unlist), 3);
	state.registerInternalFunction(state.internStr("length"), (length), 1);
	
	state.registerInternalFunction(state.internStr("eval"), (eval_fn), 3);
	state.registerInternalFunction(state.internStr("source"), (source), 1);

	state.registerInternalFunction(state.internStr("lapply"), (lapply), 2);
	//state.registerInternalFunction(state.internStr("t.list"), (tlist));

	state.registerInternalFunction(state.internStr("environment"), (environment), 1);
	state.registerInternalFunction(state.internStr("parent.frame"), (parentframe), 1);
	state.registerInternalFunction(state.internStr("sys.call"), (syscall), 1);
	state.registerInternalFunction(state.internStr("remove"), (remove), 2);
	
	state.registerInternalFunction(state.internStr("stop"), (stop_fn), 1);
	state.registerInternalFunction(state.internStr("warning"), (warning_fn), 1);
	
	state.registerInternalFunction(state.internStr("paste"), (paste), 2);
	state.registerInternalFunction(state.internStr("deparse"), (deparse), 1);
	state.registerInternalFunction(state.internStr("substitute"), (substitute), 1);
	
	state.registerInternalFunction(state.internStr("typeof"), (type_of), 1);
	
	state.registerInternalFunction(state.internStr("exists"), (exists), 4);

	state.registerInternalFunction(state.internStr("proc.time"), (proctime), 0);
	state.registerInternalFunction(state.internStr("trace.config"), (traceconfig), 1);
}

