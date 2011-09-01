
#include "internal.h"
#include "compiler.h"
#include "parser.h"
#include "library.h"
#include <math.h>
#include <fstream>

const MaxOp<TComplex>::A MaxOp<TComplex>::Base = std::complex<double>(0,0);
const MinOp<TComplex>::A MinOp<TComplex>::Base = std::complex<double>(0,0);
const AnyOp::A AnyOp::Base = 0;
const AllOp::A AllOp::Base = 1;

static void checkNumArgs(List const& args, int64_t nargs) {
	if(args.length > nargs) _error("unused argument(s)");
	else if(args.length < nargs) _error("too few arguments");
}

Value cat(State& state, List const& args, Character const& names) {
	for(int64_t i = 0; i < args.length; i++) {
		Character c = As<Character>(state, force(state, args[i]));
		for(int64_t j = 0; j < c.length; j++) {
			printf("%s", state.SymToStr(c[j]).c_str());
			if(j < c.length-1) printf(" ");
		}
	}
	return Null::Singleton();
}

Value library(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);

	Character from = As<Character>(state, force(state, args[0]));
	if(from.length > 0) {
		loadLibrary(state, state.SymToStr(from[0]));
	}
	return Null::Singleton();
}

Value rm(State& state, List const& args, Character const& names) {
	for(int64_t i = 0; i < args.length; i++) 
		if(!expression(args[i]).isSymbol() && !expression(args[i]).isCharacter()) 
			_error("rm() arguments must be symbols or character vectors");
	for(int64_t i = 0; i < args.length; i++) {
		state.frame.environment->rm(Symbol(expression(args[i])));
	}
	return Null::Singleton();
}

Value sequence(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 3);

	Value from = force(state, args[0]);
	Value by   = force(state, args[1]);
	Value len  = force(state, args[2]);

	double f = asReal1(from);
	double b = asReal1(by);
	double l = asReal1(len);

	return Sequence(f, b, l);	
}

Value repeat(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 3);
	Value from = force(state, args[0]);
	assert(args.length == 3);
	
	Value vec  = force(state, args[0]);
	Value each = force(state, args[1]);
	Value len  = force(state, args[2]);
	
	double v = asReal1(vec);
	//double e = asReal1(each);
	double l = asReal1(len);
	
	Double r(l);
	for(int64_t i = 0; i < l; i++) {
		r[i] = v;
	}
	return r;
}

Value inherits(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 3);
	Value x = force(state, args[0]);
	Character what = Character(force(state, args[1]));
	Logical which = Logical(force(state, args[2]));
	// NYI: which
	Character c = klass(state, x);
	bool inherits = false;
	for(int64_t i = 0; i < what.length && !inherits; i++) {
		for(int64_t j = 0; j < c.length && !inherits; j++) {
			if(what[i] == c[j]) inherits = true;
		}
	}
	return Logical::c(inherits);
}

Value attr(State& state, List const& args, Character const& names)
{
	checkNumArgs(args, 3);
	// NYI: exact
	Value object = force(state, args[0]);
	if(object.isObject()) {
		Character which = Character(force(state, args[1]));
		return ((Object const&)object).getAttribute(which[0]);
	}
	else {
		return Value::Nil();
	}
}

Value assignAttr(State& state, List const& args, Character const& names)
{
	checkNumArgs(args, 3);
	Value object = force(state, args[0]);
	Character which = Character(force(state, args[1]));
	if(!object.isObject()) {
		Value v;
		Object::Init(v, object);
		object = v;
	}
	((Object&)object).setAttribute(which[0], force(state, args[2]));
	return object;
}

Type::Enum cTypeCast(Value const& v, Type::Enum t)
{
	Type::Enum r;
	r = std::max(v.type, t);
	return r;
}

Value list(State& state, List const& args, Character const& names) {
	List out(args.length);
	for(int64_t i = 0; i < args.length; i++) out[i] = force(state, args[i]);
	if(names.length != args.length) {
		return out;
	} else {
		Value v;
		Object::InitWithNames(v, out, names);
		return v;
	}
}

Value unlist(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value v = force(state, args[0]);
	if(!v.isList()) {
		return v;
	}
	List from = List(v);
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
				return out; } break;
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
		for(int64_t i = 0; i < length; i++) if(-de[i] > 0 && -de[i] < (int64_t)a.length) index.insert(-de[i]);
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
#define CASE(Name,...) case Type::Name: SubsetInclude<Name>::eval(state, Name(a), index, positive, out); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};
		}
		else if(negative > 0) {
			switch(a.type) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetExclude<Name>::eval(state, Name(a), index, negative, out); break;
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
#define CASE(Name) case Type::Name: SubsetLogical<Name>::eval(state, Name(a), index, out); break;
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
struct SubsetAssignT {
	static A eval(State& state, A const& a, Integer const& d, A const& b)
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
		return r;
	}
};

void SubsetAssignSlow(State& state, Value const& a, Value const& i, Value const& b, Value& c) {
	Integer idx = As<Integer>(state, i);
	switch(a.type) {
		#define CASE(Name) case Type::Name: c = SubsetAssignT<Name>::eval(state, (Name const&)a, idx, As<Name>(state, b)); break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: subset assign type"); break;
	};
}

void Subset2AssignSlow(State& state, Value const& a, Value const& i, Value const& b, Value& c) {
	Integer idx = As<Integer>(state, i);
	switch(a.type) {
		#define CASE(Name) case Type::Name: c = SubsetAssignT<Name>::eval(state, (Name const&)a, idx, As<Name>(state, b)); break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: subset assign type"); break;
	};
}

Value length(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Integer i(1);
	i[0] = a.length;
	return i;
}

Value quote(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	return expression(args[0]);
}

Value eval_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 3);
	Value expr = force(state, args[0]);
	Value envir = force(state, args[1]);
	//Value enclos = force(state, args[2]);
	return eval(state, Compiler::compile(state, expr), REnvironment(envir).ptr());
}

Value lapply(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 2);
	List x = As<List>(state, force(state, args[0]));
	Value func = force(state, args[1]);

	List apply(2);
	apply[0] = func;

	List result(x.length);
	// TODO: should have a way to make a simple function call without compiling,
	// or should have a fast case for compilation
	for(int64_t i = 0; i < x.length; i++) {
		apply[1] = x[i];
		result[i] = eval(state, Compiler::compile(state, CreateCall(apply)));
	}

	return result;
}

Value tlist(State& state, List const& args, Character const& names) {
	int64_t length = args.length > 0 ? 1 : 0;
	List a = Clone(args);
	for(int64_t i = 0; i < a.length; i++) {
		a[i] = force(state, a[i]);
		if(a[i].isVector() && a[i].length != 0 && length != 0)
			length = std::max(length, (int64_t)a[i].length);
	}
	List result(length);
	for(int64_t i = 0; i < length; i++) {
		List element(args.length);
		for(int64_t j = 0; j < a.length; j++) {
			if(a[j].isVector())
				Element2(a[j], i%a[j].length, element[j]);
			else
				element[j] = a[j];
		}
		result[i] = element;
	}
	return result;
}

Value source(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value file = force(state, args[0]);
	std::ifstream t(state.SymToStr(Character(file)[0]).c_str());
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string code = buffer.str();

	Parser parser(state);
	Value value;
	parser.execute(code.c_str(), code.length(), true, value);	
	
	return eval(state, Compiler::compile(state, value));
}

Value switch_fn(State& state, List const& args, Character const& names) {
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
	return Null::Singleton();
}

Value environment(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value e = force(state, args[0]);
	if(e.isNull()) {
		return REnvironment(state.frame.environment);
	}
	else if(e.isFunction()) {
		return REnvironment(Function(e).environment());
	}
	return Null::Singleton();
}

Value parentframe(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	int64_t i = (int64_t)asReal1(force(state, args[0]));
	Environment* env = state.frame.environment;
	while(i > 0 && env != env->DynamicParent()) {
		env = env->DynamicParent();
		i--;
	}
	return REnvironment(env);
}

Value stop_fn(State& state, List const& args, Character const& names) {
	// this should stop whether or not the arguments are correct...
	std::string message = "user stop";
	if(args.length > 0) {
		if(args[0].isCharacter() && Character(args[0]).length > 0) {
			message = state.SymToStr(Character(args[0])[0]);
		}
	}
	_error(message);
	return Null::Singleton();
}

Value warning_fn(State& state, List const& args, Character const& names) {
	std::string message = "user warning";
	if(args.length > 0) {
		if(args[0].isCharacter() && Character(args[0]).length > 0) {
			message = state.SymToStr(Character(args[0])[0]);
		}
	}
	_warning(state, message);
	return Character::c(state.StrToSym(message));
} 

Value missing(State& state, List const& args, Character const& names) {
	Symbol s(expression(args[0])); 
	Value v =  state.frame.environment->get(s);
	return (v.isNil() || (v.isPromise() && Function(v).environment() == state.frame.environment)) ? Logical::True() : Logical::False();
}

Value max_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<FoldLeft, MaxOp>(state, a, result);
	return result;
}

Value min_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<FoldLeft, MinOp>(state, a, result);
	return result;
}

Value sum_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<FoldLeft, SumOp>(state, a, result);
	return result;
}

Value prod_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<FoldLeft, ProdOp>(state, a, result);
	return result;
}

Value cummax_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<ScanLeft, MaxOp>(state, a, result);
	return result;
}

Value cummin_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<ScanLeft, MinOp>(state, a, result);
	return result;
}

Value cumsum_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<ScanLeft, SumOp>(state, a, result);
	return result;
}

Value cumprod_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<ScanLeft, ProdOp>(state, a, result);
	return result;
}

Value any_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryLogical<FoldLeft, AnyOp>(state, a, result);
	return result;
}

Value all_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryLogical<FoldLeft, AllOp>(state, a, result);
	return result;
}

Value isna_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryFilter<Zip1, IsNAOp>(state, a, result);
	return result;
}

Value isnan_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryFilter<Zip1, IsNaNOp>(state, a, result);
	return result;
}

Value nchar_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 3);
	Value a = force(state, args[0]);
	// NYI: type or allowNA
	Value result;
	unaryCharacter<Zip1, NcharOp>(state, a, result);
	return result;
}

Value nzchar_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryCharacter<Zip1, NzcharOp>(state, a, result);
	return result;
}

Value isfinite_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryFilter<Zip1, IsFiniteOp>(state, a, result);
	return result;
}

Value isinfinite_fn(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryFilter<Zip1, IsInfiniteOp>(state, a, result);
	return result;
}

Value paste(State& state, List const& args, Character const& names) {
	Character a = As<Character>(state, force(state, args[0]));
	Character sep = As<Character>(state, force(state, args[1]));
	std::string result = "";
	for(int64_t i = 0; i+1 < a.length; i++) {
		result = result + state.SymToStr(a[i]) + state.SymToStr(sep[0]);
	}
	if(a.length > 0) result = result + state.SymToStr(a[a.length-1]);
	return Character::c(state.StrToSym(result));
}

Value deparse(State& state, List const& args, Character const& names) {
	Value v = force(state, args[0]);
	return Character::c(state.StrToSym(state.deparse(v)));
}

Value substitute(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	Value v = args[0];
	while(v.isPromise()) v = Function(v).prototype()->expression;
	
	if(v.isSymbol()) {
		Value r = state.frame.environment->get(Symbol(v));
		if(!r.isNil()) v = r;
		while(v.isPromise()) v = Function(v).prototype()->expression;
	}
 	return v;
}

Value type_of(State& state, List const& args, Character const& names) {
	// Should have a direct mapping from type to symbol.
	Value v = force(state, args[0]);
	return Character::c(state.StrToSym(Type::toString(v.type)));
}

Value get(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 2);
	Character c = As<Character>(state, force(state, args[0]));
	REnvironment e(force(state, args[1]));
	Value v = e.ptr()->get(c[0]);
	if(v.isNil())
		return Null::Singleton();
	else
		return v;
}

Value exists(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 2);
	Character c = As<Character>(state, force(state, args[0]));
	REnvironment e(force(state, args[1]));
	Value v = e.ptr()->get(c[0]);
	if(v.isNil())
		return Logical::False();
	else
		return Logical::True();
}

#include <sys/time.h>

uint64_t readTime()
{
  timeval time_tt;
  gettimeofday(&time_tt, NULL);
  return (uint64_t)time_tt.tv_sec * 1000 * 1000 + (uint64_t)time_tt.tv_usec;
}

Value systemtime(State& state, List const& args, Character const& names) {
	checkNumArgs(args, 1);
	uint64_t s = readTime();
	force(state, args[0]);
	uint64_t e = readTime();
	return Double::c((e-s)/(1000000.0));
}

void importCoreFunctions(State& state, Environment* env)
{
	env->assign(state.StrToSym("max"), BuiltIn(max_fn));
	env->assign(state.StrToSym("min"), BuiltIn(min_fn));
	env->assign(state.StrToSym("sum"), BuiltIn(sum_fn));
	env->assign(state.StrToSym("prod"), BuiltIn(prod_fn));
	env->assign(state.StrToSym("cummax"), BuiltIn(cummax_fn));
	env->assign(state.StrToSym("cummin"), BuiltIn(cummin_fn));
	env->assign(state.StrToSym("cumsum"), BuiltIn(cumsum_fn));
	env->assign(state.StrToSym("cumprod"), BuiltIn(cumprod_fn));
	env->assign(state.StrToSym("any"), BuiltIn(any_fn));
	env->assign(state.StrToSym("all"), BuiltIn(all_fn));
	env->assign(state.StrToSym("nchar"), BuiltIn(nchar_fn));
	env->assign(state.StrToSym("nzchar"), BuiltIn(nzchar_fn));
	env->assign(state.StrToSym("is.na"), BuiltIn(isna_fn));
	env->assign(state.StrToSym("is.nan"), BuiltIn(isnan_fn));
	env->assign(state.StrToSym("is.finite"), BuiltIn(isfinite_fn));
	env->assign(state.StrToSym("is.infinite"), BuiltIn(isinfinite_fn));
	
	env->assign(state.StrToSym("cat"), BuiltIn(cat));
	env->assign(state.StrToSym("library"), BuiltIn(library));
	env->assign(state.StrToSym("rm"), BuiltIn(rm));
	env->assign(state.StrToSym("inherits"), BuiltIn(inherits));
	
	env->assign(state.StrToSym("seq"), BuiltIn(sequence));
	env->assign(state.StrToSym("rep"), BuiltIn(repeat));
	
	env->assign(state.StrToSym("attr"), BuiltIn(attr));
	env->assign(state.StrToSym("attr<-"), BuiltIn(assignAttr));
	
	env->assign(state.StrToSym("list"), BuiltIn(list));
	env->assign(state.StrToSym("unlist"), BuiltIn(unlist));
	env->assign(state.StrToSym("length"), BuiltIn(length));
	
	env->assign(state.StrToSym("switch"), BuiltIn(switch_fn));

	env->assign(state.StrToSym("eval"), BuiltIn(eval_fn));
	env->assign(state.StrToSym("quote"), BuiltIn(quote));
	env->assign(state.StrToSym("source"), BuiltIn(source));

	env->assign(state.StrToSym("lapply"), BuiltIn(lapply));
	env->assign(state.StrToSym("t.list"), BuiltIn(tlist));

	env->assign(state.StrToSym("environment"), BuiltIn(environment));
	env->assign(state.StrToSym("parent.frame"), BuiltIn(parentframe));
	env->assign(state.StrToSym("missing"), BuiltIn(missing));
	
	env->assign(state.StrToSym("stop"), BuiltIn(stop_fn));
	env->assign(state.StrToSym("warning"), BuiltIn(warning_fn));
	
	env->assign(state.StrToSym("paste"), BuiltIn(paste));
	env->assign(state.StrToSym("deparse"), BuiltIn(deparse));
	env->assign(state.StrToSym("substitute"), BuiltIn(substitute));
	
	env->assign(state.StrToSym("typeof"), BuiltIn(type_of));
	
	env->assign(state.StrToSym("get"), BuiltIn(get));
	env->assign(state.StrToSym("exists"), BuiltIn(exists));

	env->assign(state.StrToSym("system.time"), BuiltIn(systemtime));
}

