
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

void cat(State& state, Value const* args, Value& result) {
	List const& a = Cast<List>(args[0]);
	for(int64_t i = 0; i < a.length; i++) {
		if(!List::isNA(a[i])) {
			Character c = As<Character>(state, a[i]);
			for(int64_t j = 0; j < c.length; j++) {
				printf("%s", state.externStr(c[j]).c_str());
				if(j < c.length-1) printf(" ");
			}
		}
	}
	result = Null::Singleton();
}

void remove(State& state, Value const* args, Value& result) {
	Character const& a = Cast<Character>(args[0]);
	REnvironment e(args[1]);
	for(int64_t i = 0; i < a.length; i++) {
		e.ptr()->remove(a[i]);
	}
	result = Null::Singleton();
}

void library(State& state, Value const* args, Value& result) {
	Character from = As<Character>(state, args[0]);
	if(from.length > 0) {
		loadLibrary(state, "library", state.externStr(from[0]));
	}
	result = Null::Singleton();
}

void sequence(State& state, Value const* args, Value& result) {
	double from = asReal1(args[0]);
	double by = asReal1(args[1]);
	double len = asReal1(args[2]);

	result = Sequence(from, by, len);	
}

void repeat(State& state, Value const* args, Value& result) {
	double v = asReal1(args[0]);
	//double e = asReal1(args[1]);
	double l = asReal1(args[2]);
	
	Double r(l);
	for(int64_t i = 0; i < l; i++) {
		r[i] = v;
	}
	result = r;
}

void inherits(State& state, Value const* args, Value& result) {
	Value x = args[0];
	Character what = Cast<Character>(args[1]);
	Logical which = Cast<Logical>(args[2]);
	// NYI: which
	Character c = klass(state, x);
	bool inherits = false;
	for(int64_t i = 0; i < what.length && !inherits; i++) {
		for(int64_t j = 0; j < c.length && !inherits; j++) {
			if(what[i] == c[j]) inherits = true;
		}
	}
	result = Logical::c(inherits);
}

void attr(State& state, Value const* args, Value& result)
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

void assignAttr(State& state, Value const* args, Value& result)
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
int64_t unlistLength(State& state, int64_t recurse, Value a) {
	if(a.isObject()) a = ((Object&)a).base();
	if(recurse > 0 && a.isList()) {
		List l(a);
		int64_t t = 0;
		for(int64_t i = 0; i < l.length; i++) 
			t += unlistLength(state, recurse-1, l[i]);
		return t;
	}
	else if(a.isVector()) return a.length;
	else return 1;
}

Type::Enum unlistType(State& state, int64_t recurse, Value a) {
	if(a.isObject()) a = ((Object&)a).base();
	if(a.isList()) {
		List l(a);
		Type::Enum t = Type::Null;
		for(int64_t i = 0; i < l.length; i++) 
			t = cTypeCast(recurse > 0 ? unlistType(state, recurse-1, l[i]) : l[i].type, t);
		return t;
	}
	else if(a.isVector()) return a.type;
	else return Type::List;
}

template< class T >
void unlist(State& state, int64_t recurse, Value a, T& out, int64_t& start) {
	if(a.isObject()) a = ((Object&)a).base();
	if(recurse > 0 && a.isList()) {
		List l(a);
		for(int64_t i = 0; i < l.length; i++) 
			unlist(state, recurse-1, l[i], out, start);
		return;
	}
	else if(a.isVector()) { Insert(state, a, 0, out, start, a.length); start += a.length; }
	else _error("Unexpected non-basic type in unlist");
}

template<>
void unlist<List>(State& state, int64_t recurse, Value a, List& out, int64_t& start) {
	if(a.isObject()) a = ((Object&)a).base();
	if(recurse > 0 && a.isList()) {
		List l(a);
		for(int64_t i = 0; i < l.length; i++) 
			unlist(state, recurse-1, l[i], out, start);
		return;
	}
	else if(a.isVector()) { Insert(state, a, 0, out, start, a.length); start += a.length; }
	else out[start++] = a;
}

bool unlistHasNames(State& state, int64_t recurse, Value a) {
	if(a.isObject() && ((Object const&)a).hasNames()) return true;
	if(a.isObject()) a = ((Object&)a).base();
	if(recurse > 0 && a.isList()) {
		List l(a);
		bool hasNames = false;
		for(int64_t i = 0; i < l.length; i++) 
			hasNames = hasNames || unlistHasNames(state, recurse-1, l[i]);
		return hasNames;
	}
	else return false;
}

std::string makeName(State& state, std::string prefix, String name, int64_t i) {
	if(prefix.length() > 0) {
		if(name != Strings::empty)
			return prefix + "." + state.externStr(name);
		else
			return prefix + intToStr(i+1);
	}
	else {
		return state.externStr(name);
	}
}

void unlistNames(State& state, int64_t recurse, Value a, Character& out, int64_t& start, std::string prefix) {
	Character names(0);
	if(a.isObject() && ((Object&)a).hasNames()) {
		names = (Character)((Object&)a).getNames();
	}
	if(a.isObject()) a = ((Object&)a).base();

	if(recurse > 0 && a.isList()) {
		List l(a);
		for(int64_t i = 0; i < l.length; i++)
			unlistNames(state, recurse-1, l[i], out, start, makeName(state, prefix, i < names.length ? names[i] : Strings::empty, i));
		return;
	}
	else if(a.isVector() && a.length != 1) { 
		for(int64_t i = 0; i < a.length; i++) {
			out[start++] = state.internStr(makeName(state, prefix, (i < names.length) ? names[i] : Strings::empty, i));
		}
	}
	else out[start++] = state.internStr(prefix);
}

// TODO: useNames parameter could be handled at the R level
void unlist(State& state, Value const* args, Value& result) {
	Value v = args[0];
	int64_t recurse = Cast<Logical>(args[1])[0] ? std::numeric_limits<int64_t>::max() : 1;
	bool useNames = Cast<Logical>(args[2])[0];
	
	int64_t length = unlistLength(state, recurse, v);
	Type::Enum type = unlistType(state, recurse, v);

	switch(type) {
		#define CASE(Name) \
			case Type::Name: { \
				Name out(length); \
				int64_t i = 0; \
				unlist(state, recurse, v, out, i); \
				result = out; \
			} break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Insert into this type"); break;
	};

	if(useNames && unlistHasNames(state, recurse, v)) {
		Character outnames(length);
		int64_t i = 0;
		unlistNames(state, recurse, v, outnames, i, "");
		Object::Init(result, result, outnames);
	}
}


template< class A >
struct SubsetInclude {
	static void eval(State& state, A const& a, Integer const& d, int64_t nonzero, Value& out)
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
	static void eval(State& state, A const& a, Integer const& d, int64_t nonzero, Value& out)
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
	static void eval(State& state, A const& a, Logical const& d, Value& out)
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
inline int64_t find(State& state, A const& a, typename A::Element const& b) {
	typename A::Element const* ae = a.v();
	// eventually use an index here...
	for(int64_t i = 0; i < a.length; i++) {
		if(ae[i] == b) return i;
	}
	return -1;
}

template< class A, class B >
struct SubsetIndexed {
	static void eval(State& state, A const& a, B const& b, B const& d, Value& out)
	{
		typename A::Element const* ae = a.v();
		typename B::Element const* de = d.v();
		
		int64_t length = d.length;
		A r(length);
		for(int64_t i = 0; i < length; i++) {
			int64_t index = find(state, b, de[i]);
			r[i] = index >= 0 ? ae[index] : A::NAelement;
		}
		out = r;
	}
};

void SubsetSlow(State& state, Value const& a, Value const& i, Value& out) {
	if(i.isDouble() || i.isInteger()) {
		if(a.isObject()) {
			// TODO: this computes positive twice
			Value r, names;
			SubsetSlow(state, ((Object const&)a).base(), i, r);
			if(((Object const&)a).hasNames()) {
				SubsetSlow(state, ((Object const&)a).getNames(), i, names);
				Object::Init(r, r, names);
			}
			out = r;
			return;
		}
		Integer index = As<Integer>(state, i);
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
#define CASE(Name,...) case Type::Name: SubsetInclude<Name>::eval(state, (Name const&)a, index, positive, out); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};
		}
		else if(negative > 0) {
			switch(a.type) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetExclude<Name>::eval(state, (Name const&)a, index, negative, out); break;
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
			SubsetSlow(state, ((Object const&)a).base(), i, r);
			if(((Object const&)a).hasNames()) {
				SubsetSlow(state, ((Object const&)a).getNames(), i, names);
				Object::Init(r, r, names);
			}
			out = r;
			return;
		}
		Logical index = Logical(i);
		switch(a.type) {
			case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetLogical<Name>::eval(state, (Name const&)a, index, out); break;
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
#define CASE(Name) case Type::Name: SubsetIndexed<Name, Character>::eval(state, (Name const&)b, (Character const&)n, (Character const&)i, out); break;
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
	static void eval(State& state, A const& a, bool clone, Integer const& d, A const& b, Value& out)
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
		Resize(state, clone, r, outlength);
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
	static void eval(State& state, A const& a, bool clone, Logical const& d, A const& b, Value& out)
	{
		typename A::Element const* be = b.v();
		typename Logical::Element const* de = d.v();
		
		// determine length
		int64_t length = std::max(a.length, d.length);
		A r = a;
		Resize(state, clone, r, length);
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
	static void eval(State& state, A const& a, bool clone, B const& b, B const& d, A const& v, Value& out)
	{
		typename B::Element const* de = d.v();
		
		std::map<typename B::Element, int64_t> overflow;
		int64_t extra = a.length;

		int64_t length = d.length;
		Integer include(length);
		for(int64_t i = 0; i < length; i++) {
			int64_t index = find(state, b, de[i]);
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
		SubsetAssignInclude<A>::eval(state, a, clone, include, v, out);
	}
};

void SubsetAssignSlow(State& state, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(i.isDouble() || i.isInteger()) {
		Integer idx = As<Integer>(state, i);
		switch(a.type) {
#define CASE(Name) case Type::Name: SubsetAssignInclude<Name>::eval(state, (Name const&)a, clone, idx, As<Name>(state, b), c); break;
			VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			case Type::Object: {
				Value r;
				SubsetAssignSlow(state, ((Object const&)a).base(), clone, i, b, r);
				if(((Object const&)a).hasNames()) {
					Character names = (Character)((Object const&)a).getNames();
					Resize(state, clone, names, r.length, Strings::empty);
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
#define CASE(Name) case Type::Name: SubsetAssignLogical<Name>::eval(state, (Name const&)a, clone, index, As<Name>(state, b), c); break;
					 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			case Type::Object: {
				Value r;
				SubsetAssignSlow(state, ((Object const&)a).base(), clone, i, b, r);
				if(((Object const&)a).hasNames()) {
					Character names = (Character)((Object const&)a).getNames();
					Resize(state, clone, names, r.length, Strings::empty);
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
#define CASE(Name) case Type::Name: SubsetAssignIndexed<Name, Character>::eval(state, (Name const&)base, clone, (Character const&)names, (Character const&)i, As<Name>(state, b), c); break;
				VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};
			Value newNames;
			SubsetAssignIndexed<Character, Character>::eval(state, (Character const&)names, clone, (Character const&)names, (Character const&)i, (Character const&)i, newNames);
			Object::Init(c, c, (Character const&)newNames);
		}
		else _error("Object does not have names for subsetting");
	}
	else {
		_error("NYI indexing type");
	}
}

void Subset2AssignSlow(State& state, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(i.isDouble() || i.isInteger() || i.isCharacter())
		SubsetAssignSlow(state, a, clone, i, b, c);
	// Frankly it seems pointless to support this case. We should make this an error.
	//else if(i.isLogical() && i.length == 1 && ((Logical const&)i)[0])
	//	SubsetAssignSlow(state, a, clone, As<Integer>(state, i), b, c);
	else
		_error("NYI indexing type");
}

void length(State& state, Value const* args, Value& result) {
	result = Integer::c(args[0].length);
}

void eval_fn(State& state, Value const* args, Value& result) {
	result = eval(state, Compiler::compile(state, args[0]), REnvironment(args[1]).ptr());
}

struct lapplyargs {
	uint64_t start;
	uint64_t end;
	State& state;
	List& in;
	List& out;
	Value func;
};

void* lapplybody(void* args) {
	lapplyargs& l = *(lapplyargs*)args;
	List apply(2);
	apply[0] = l.func;
	apply[1] = Value::Nil();
	Prototype* p = Compiler::compile(l.state, CreateCall(apply));
	State istate(l.state.sharedState);
	istate.tracing.config = l.state.tracing.config;
	istate.tracing.verbose = l.state.tracing.verbose;
	for( size_t i=l.start; i!=l.end; ++i ) {
		p->calls[0].arguments[0] = l.in[i];
		l.out[i] = eval(istate, p);
	}
	return 0;
}

void lapply(State& state, Value const* args, Value& result) {
	List x = As<List>(state, args[0]);
	Value func = args[1];
	List r(x.length);

	/*List apply(2);
	apply[0] = func;

	// TODO: should have a way to make a simple function call without compiling,
	// or should have a fast case for compilation
	State istate(state.sharedState);
	for(int64_t i = 0; i < x.length; i++) {
		apply[1] = x[i];
		r[i] = eval(istate, Compiler::compile(state, CreateCall(apply)));
	}*/

	/*List apply(2);
	apply[0] = func;
	apply[1] = Value::Nil();
	Prototype* p = Compiler::compile(state, CreateCall(apply));
	State istate(state.sharedState);
	for(int64_t i = 0; i < x.length; i++) {
		p->calls[0].arguments[0] = x[i];
		r[i] = eval(istate, p);
	}*/

	pthread_t h1, h2;

	lapplyargs a1 = (lapplyargs) {0, x.length/2, state, x, r, func};
	lapplyargs a2 = (lapplyargs) {x.length/2, x.length, state, x, r, func};

        pthread_create (&h1, NULL, lapplybody, &a1);
        pthread_create (&h2, NULL, lapplybody, &a2);
	pthread_join(h1, NULL);
	pthread_join(h2, NULL);

	result = r;
}

/*
void tlist(State& state, Value const* args, Value& result) {
	int64_t length = args.length > 0 ? 1 : 0;
	List a = Clone(args);
	for(int64_t i = 0; i < a.length; i++) {
		a[i] = force(state, a[i]);
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
void source(State& state, Value const* args, Value& result) {
	Character file = Cast<Character>(args[0]);
	std::ifstream t(state.externStr(file[0]).c_str());
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string code = buffer.str();

	Parser parser(state);
	Value value;
	parser.execute(code.c_str(), code.length(), true, value);	
	
	result = eval(state, Compiler::compile(state, value));
}

void environment(State& state, Value const* args, Value& result) {
	Value e = args[0];
	if(e.isNull()) {
		result = REnvironment(state.frame.environment);
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
void parentframe(State& state, Value const* args, Value& result) {
	int64_t i = (int64_t)asReal1(args[0]);
	Environment* env = state.frame.environment;
	while(i > 0 && env->DynamicScope() != 0) {
		env = env->DynamicScope();
		i--;
	}
	result = REnvironment(env);
}

void syscall(State& state, Value const* args, Value& result) {
	int64_t i = (int64_t)asReal1(args[0]);
	Environment* env = state.frame.environment;
	while(i > 0 && env->DynamicScope() != 0) {
		env = env->DynamicScope();
		i--;
	}
	result = env->call;
}

void stop_fn(State& state, Value const* args, Value& result) {
	// this should stop whether or not the arguments are correct...
	std::string message = state.externStr(Cast<Character>(args[0])[0]);
	_error(message);
	result = Null::Singleton();
}

void warning_fn(State& state, Value const* args, Value& result) {
	std::string message = state.externStr(Cast<Character>(args[0])[0]);
	_warning(state, message);
	result = Character::c(state.internStr(message));
} 
void isna_fn(State& state, Value const* args, Value& result) {
	unaryFilter<Zip1, IsNAOp>(state, args[0], result);
}

void isnan_fn(State& state, Value const* args, Value& result) {
	unaryFilter<Zip1, IsNaNOp>(state, args[0], result);
}

void nchar_fn(State& state, Value const* args, Value& result) {
	// NYI: type or allowNA
	unaryCharacter<Zip1, NcharOp>(state, args[0], result);
}

void nzchar_fn(State& state, Value const* args, Value& result) {
	unaryCharacter<Zip1, NzcharOp>(state, args[0], result);
}

void isfinite_fn(State& state, Value const* args, Value& result) {
	unaryFilter<Zip1, IsFiniteOp>(state, args[0], result);
}

void isinfinite_fn(State& state, Value const* args, Value& result) {
	unaryFilter<Zip1, IsInfiniteOp>(state, args[0], result);
}

void paste(State& state, Value const* args, Value& result) {
	Character a = As<Character>(state, args[0]);
	Character sep = As<Character>(state, args[1]);
	std::string r = "";
	for(int64_t i = 0; i+1 < a.length; i++) {
		r = r + state.externStr(a[i]) + state.externStr(sep[0]);
	}
	if(a.length > 0) r = r + state.externStr(a[a.length-1]);
	result = Character::c(state.internStr(r));
}

void deparse(State& state, Value const* args, Value& result) {
	result = Character::c(state.internStr(state.deparse(args[0])));
}

void substitute(State& state, Value const* args, Value& result) {
	Value v = args[0];
	while(v.isPromise()) v = Function(v).prototype()->expression;
	
	if(v.isSymbol()) {
		Value r = state.frame.environment->get(Symbol(v));
		if(!r.isNil()) v = r;
		while(v.isPromise()) v = Function(v).prototype()->expression;
	}
 	result = v;
}

void type_of(State& state, Value const* args, Value& result) {
	// Should have a direct mapping from type to symbol.
	result = Character::c(state.internStr(Type::toString(args[0].type)));
}

void exists(State& state, Value const* args, Value& result) {
	Character c = As<Character>(state, args[0]);
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

void proctime(State& state, Value const* args, Value& result) {
	uint64_t s = readTime();
	result = Double::c(s/(1000000.0));
}

void traceconfig(State & state, Value const* args, Value& result) {
	Integer c = As<Integer>(state, args[0]);
	if(c.length == 0) _error("condition is of zero length");
	state.tracing.config = (TraceState::Mode) c[0];
	result = Null::Singleton();
}

void importCoreFunctions(State& state, Environment* env)
{
	state.sharedState.registerInternalFunction(state.internStr("nchar"), (nchar_fn), 1);
	state.sharedState.registerInternalFunction(state.internStr("nzchar"), (nzchar_fn), 1);
	state.sharedState.registerInternalFunction(state.internStr("is.na"), (isna_fn), 1);
	state.sharedState.registerInternalFunction(state.internStr("is.nan"), (isnan_fn), 1);
	state.sharedState.registerInternalFunction(state.internStr("is.finite"), (isfinite_fn), 1);
	state.sharedState.registerInternalFunction(state.internStr("is.infinite"), (isinfinite_fn), 1);
	
	state.sharedState.registerInternalFunction(state.internStr("cat"), (cat), 1);
	state.sharedState.registerInternalFunction(state.internStr("library"), (library), 1);
	state.sharedState.registerInternalFunction(state.internStr("inherits"), (inherits), 3);
	
	state.sharedState.registerInternalFunction(state.internStr("seq"), (sequence), 3);
	state.sharedState.registerInternalFunction(state.internStr("rep"), (repeat), 3);
	
	state.sharedState.registerInternalFunction(state.internStr("attr"), (attr), 3);
	state.sharedState.registerInternalFunction(state.internStr("attr<-"), (assignAttr), 3);
	
	state.sharedState.registerInternalFunction(state.internStr("unlist"), (unlist), 3);
	state.sharedState.registerInternalFunction(state.internStr("length"), (length), 1);
	
	state.sharedState.registerInternalFunction(state.internStr("eval"), (eval_fn), 3);
	state.sharedState.registerInternalFunction(state.internStr("source"), (source), 1);

	state.sharedState.registerInternalFunction(state.internStr("lapply"), (lapply), 2);
	//state.sharedState.registerInternalFunction(state.internStr("t.list"), (tlist));

	state.sharedState.registerInternalFunction(state.internStr("environment"), (environment), 1);
	state.sharedState.registerInternalFunction(state.internStr("parent.frame"), (parentframe), 1);
	state.sharedState.registerInternalFunction(state.internStr("sys.call"), (syscall), 1);
	state.sharedState.registerInternalFunction(state.internStr("remove"), (remove), 2);
	
	state.sharedState.registerInternalFunction(state.internStr("stop"), (stop_fn), 1);
	state.sharedState.registerInternalFunction(state.internStr("warning"), (warning_fn), 1);
	
	state.sharedState.registerInternalFunction(state.internStr("paste"), (paste), 2);
	state.sharedState.registerInternalFunction(state.internStr("deparse"), (deparse), 1);
	state.sharedState.registerInternalFunction(state.internStr("substitute"), (substitute), 1);
	
	state.sharedState.registerInternalFunction(state.internStr("typeof"), (type_of), 1);
	
	state.sharedState.registerInternalFunction(state.internStr("exists"), (exists), 4);

	state.sharedState.registerInternalFunction(state.internStr("proc.time"), (proctime), 0);
	state.sharedState.registerInternalFunction(state.internStr("trace.config"), (traceconfig), 1);
}

