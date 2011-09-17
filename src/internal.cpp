
#include "internal.h"
#include "compiler.h"
#include "parser.h"
#include "library.h"
#include <math.h>
#include <fstream>

template<typename T>
T const& Cast(Value const& v) {
	if(v.type != T::VectorType) _error("incorrect type passed to internal function");
	return (T const&)v;
}

void cat(State& state, Value const* args, Value& result) {
	List const& a = Cast<List>(args[0]);
	for(int64_t i = 0; i < a.length; i++) {
		if(!List::isNA(a[i])) {
			Character c = As<Character>(state, force(state, a[i]));
			for(int64_t j = 0; j < c.length; j++) {
				printf("%s", state.externStr(c[j]).c_str());
				if(j < c.length-1) printf(" ");
			}
		}
	}
	result = Null::Singleton();
}

void library(State& state, Value const* args, Value& result) {
	Character from = As<Character>(state, args[0]);
	if(from.length > 0) {
		loadLibrary(state, state.externStr(from[0]));
	}
	result = Null::Singleton();
}

void remove(State& state, Value const* args, Value& result) {
	List const& list = Cast<List>(args[0]);
	for(int64_t i = 0; i < list.length; i++) 
		if(!expression(list[i]).isSymbol() && !expression(list[i]).isCharacter()) 
			_error("rm() arguments must be symbols or character vectors");
	for(int64_t i = 0; i < list.length; i++) {
		state.frame.environment->rm(Symbol(expression(list[i])));
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
	((Object&)object).setAttribute(which[0], args[2]);
	result = object;
}

Type::Enum cTypeCast(Value const& v, Type::Enum t)
{
	Type::Enum r;
	r = std::max(v.type, t);
	return r;
}
/*
void list(State& state, Value const* args, Value& result) {
	List out(args.length);
	for(int64_t i = 0; i < args.length; i++) out[i] = force(state, args[i]);
	if(names.length == 0) {
		return out;
	} else {
		Value v;
		Object::InitWithNames(v, out, names);
		return v;
	}
}
*/
void unlist(State& state, Value const* args, Value& result) {
	Value v = args[0];
	if(!v.isList()) {
		result = v;
		return;
	}
	List from = Cast<List>(v);
	int64_t total = 0;
	Type::Enum type = Type::Null;
	for(int64_t i = 0; i < from.length; i++) {
		from[i] = force(state, from[i]);
		total += from[i].length;
		type = cTypeCast(from[i], type);
	}
	Character outnames;
	/*if(hasNames(from))
	{
		Character names = Character(getNames(from));
		outnames = Character(total);
		int64_t j = 0;
		for(int64_t i = 0; i < (int64_t)from.length; i++) {
			for(int64_t m = 0; m < from[i].length; m++, j++) {
				// NYI: R makes these names distinct
				outnames[j] = names[i];
			}
		}
	}*/
	int64_t j = 0;
	switch(type) {
		#define CASE(Name) \
			case Type::Name: { \
				Name out(total); \
				for(int64_t i = 0; i < from.length; i++) { \
					Insert(state, from[i], 0, out, j, from[i].length); \
					j += from[i].length; \
				} \
				/*if(hasNames(from)) setNames(out, outnames);*/ \
				result = out; } break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Insert into this type"); break;
	};
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

void SubsetSlow(State& state, Value const& a, Value const& i, Value& out) {
	if(i.isDouble() || i.isInteger()) {
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
		Logical index = Logical(i);
		switch(a.type) {
			case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetLogical<Name>::eval(state, (Name const&)a, index, out); break;
					 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
		};	
	}
	else {
		_error("NYI indexing type");
	}
}

template< class A  >
struct SubsetAssignInclude {
	static void eval(State& state, A const& a, Integer const& d, A const& b, Value& out)
	{
		typename A::Element const* be = b.v();
		typename Integer::Element const* de = d.v();

		// compute max index 
		int64_t outlength = 0;
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) {
			outlength = std::max((int64_t)outlength, de[i]);
		}

		// should use max index here to extend vector if necessary	
		//A r = Clone(a);	
		A r = a;
		typename A::Element* re = r.v();
		for(int64_t i = 0; i < length; i++) {	
			int64_t idx = de[i];
			if(idx != 0)
				re[idx-1] = be[i];
		}
		out = r;
	}
};

template< class A >
struct SubsetAssignLogical {
	static void eval(State& state, A const& a, Logical const& d, A const& b, Value& out)
	{
		typename A::Element const* be = b.v();
		typename Logical::Element const* de = d.v();
		
		// determine length
		int64_t length = std::max(a.length, d.length);
		//A r = Clone(a);
		A r = a;
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

void SubsetAssignSlow(State& state, Value const& a, Value const& i, Value const& b, Value& c) {
	if(i.isDouble() || i.isInteger()) {
		Integer idx = As<Integer>(state, i);
		switch(a.type) {
#define CASE(Name) case Type::Name: SubsetAssignInclude<Name>::eval(state, (Name const&)a, idx, As<Name>(state, b), c); break;
			VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error("NYI: subset assign type"); break;
		};
	}
	else if(i.isLogical()) {
		Logical index = Logical(i);
		switch(a.type) {
#define CASE(Name) case Type::Name: SubsetAssignLogical<Name>::eval(state, (Name const&)a, index, As<Name>(state, b), c); break;
					 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
		};	
	}
	else {
		_error("NYI indexing type");
	}
}

void Subset2AssignSlow(State& state, Value const& a, Value const& i, Value const& b, Value& c) {
	Integer idx = As<Integer>(state, i);
	switch(a.type) {
		#define CASE(Name) case Type::Name: SubsetAssignInclude<Name>::eval(state, (Name const&)a, idx, As<Name>(state, b), c); break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: subset assign type"); break;
	};
}

void length(State& state, Value const* args, Value& result) {
	result = Integer::c(args[0].length);
}

void quote(State& state, Value const* args, Value& result) {
	// TODO: make op
	result = expression(args[0]);
}

void eval_fn(State& state, Value const* args, Value& result) {
	result = eval(state, Compiler::compile(state, args[0]), REnvironment(args[1]).ptr());
}

void lapply(State& state, Value const* args, Value& result) {
	List x = As<List>(state, args[0]);
	Value func = args[1];

	List apply(2);
	apply[0] = func;

	List r(x.length);
	// TODO: should have a way to make a simple function call without compiling,
	// or should have a fast case for compilation
	for(int64_t i = 0; i < x.length; i++) {
		apply[1] = x[i];
		r[i] = eval(state, Compiler::compile(state, CreateCall(apply)));
	}

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

/*
void switch_fn(State& state, Value const* args, Value& result) {
	Value one = force(state, args[0]);
	if(one.isInteger() && Integer(one).length == 1) {
		int64_t i = Integer(one)[0];
		if(i >= 1 && (int64_t)i <= args.length) {return force(state, args[i]);}
	} else if(one.isDouble() && Double(one).length == 1) {
		int64_t i = (int64_t)Double(one)[0];
		if(i >= 1 && (int64_t)i <= args.length) {return force(state, args[i]);}
	} else if(one.isCharacter() && Character(one).length == 1 && args.isObject() && ((Object const&)args).hasNames()) {
		Character names = Character(((Object const&)args).getNames());
		for(int64_t i = 1; i < args.length; i++) {
			if(names[i] == Character(one)[0]) {
				while(args[i].isNil() && i < args.length) i++;
				return i < args.length ? force(state, args[i]) : (Value)(Null::Singleton());
			}
		}
		for(int64_t i = 1; i < args.length; i++) {
			if(names[i] == Symbols::empty) {
				return force(state, args[i]);
			}
		}
	}
	result = Null::Singleton();
}
*/

void environment(State& state, Value const* args, Value& result) {
	Value e = args[0];
	if(e.isNull()) {
		result = REnvironment(state.frame.environment);
	}
	else if(e.isFunction()) {
		result = REnvironment(Function(e).environment());
	}
	result = Null::Singleton();
}

void parentframe(State& state, Value const* args, Value& result) {
	int64_t i = (int64_t)asReal1(args[0]);
	Environment* env = state.frame.environment;
	if(i > 0) {
		env = state.stack[std::max(0ULL, (unsigned long long) state.stack.size()-i)].environment;
	}
	result = REnvironment(env);
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
/*
void missing(State& state, Value const* args, Value& result) {
	Symbol s(expression(args[0])); 
	Value v =  state.frame.environment->get(s);
	result = (v.isNil() || (v.isPromise() && Function(v).environment() == state.frame.environment)) ? Logical::True() : Logical::False();
}
*/
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
	Value v = force(state, args[0]);
	result = Character::c(state.internStr(state.deparse(v)));
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
	Logical e = As<Logical>(state, args[0]);
	if(e.length == 0) _error("condition is of zero length");
	Logical v = As<Logical>(state, args[1]);
	if(v.length == 0) _error("condition is of zero length");
	state.tracing.enabled = e[0];
	state.tracing.verbose = v[0];
	result = Null::Singleton();
}

void importCoreFunctions(State& state, Environment* env)
{
	state.registerInternalFunction(state.internStr("nchar"), (nchar_fn), 1);
	state.registerInternalFunction(state.internStr("nzchar"), (nzchar_fn), 1);
	state.registerInternalFunction(state.internStr("is.na"), (isna_fn), 1);
	state.registerInternalFunction(state.internStr("is.nan"), (isnan_fn), 1);
	state.registerInternalFunction(state.internStr("is.finite"), (isfinite_fn), 1);
	state.registerInternalFunction(state.internStr("is.infinite"), (isinfinite_fn), 1);
	
	state.registerInternalFunction(state.internStr("cat"), (cat), 1);
	state.registerInternalFunction(state.internStr("library"), (library), 1);
	state.registerInternalFunction(state.internStr("remove"), (remove), 1);
	state.registerInternalFunction(state.internStr("inherits"), (inherits), 1);
	
	state.registerInternalFunction(state.internStr("seq"), (sequence), 3);
	state.registerInternalFunction(state.internStr("rep"), (repeat), 3);
	
	state.registerInternalFunction(state.internStr("attr"), (attr), 3);
	state.registerInternalFunction(state.internStr("attr<-"), (assignAttr), 3);
	
	//state.registerInternalFunction(state.internStr("list"), (list));
	state.registerInternalFunction(state.internStr("unlist"), (unlist), 1);
	state.registerInternalFunction(state.internStr("length"), (length), 1);
	
	state.registerInternalFunction(state.internStr("eval"), (eval_fn), 2);
	state.registerInternalFunction(state.internStr("quote"), (quote), 1);
	state.registerInternalFunction(state.internStr("source"), (source), 1);

	state.registerInternalFunction(state.internStr("lapply"), (lapply), 2);
	//state.registerInternalFunction(state.internStr("t.list"), (tlist));

	state.registerInternalFunction(state.internStr("environment"), (environment), 1);
	state.registerInternalFunction(state.internStr("parent.frame"), (parentframe), 1);
	//state.registerInternalFunction(state.internStr("missing"), (missing), 1);
	
	state.registerInternalFunction(state.internStr("stop"), (stop_fn), 1);
	state.registerInternalFunction(state.internStr("warning"), (warning_fn), 1);
	
	state.registerInternalFunction(state.internStr("paste"), (paste), 2);
	state.registerInternalFunction(state.internStr("deparse"), (deparse), 1);
	state.registerInternalFunction(state.internStr("substitute"), (substitute), 1);
	
	state.registerInternalFunction(state.internStr("typeof"), (type_of), 1);
	
	state.registerInternalFunction(state.internStr("exists"), (exists), 2);

	state.registerInternalFunction(state.internStr("proc.time"), (proctime), 0);
	state.registerInternalFunction(state.internStr("trace.config"), (traceconfig), 2);
}

