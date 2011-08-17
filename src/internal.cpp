
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

Value cat(State& state, List const& args) {
	for(int64_t i = 0; i < args.length; i++) {
		Character c = As<Character>(state, force(state, args[i]));
		for(int64_t j = 0; j < c.length; j++) {
			printf("%s", state.SymToStr(c[j]).c_str());
			if(j < c.length-1) printf(" ");
		}
	}
	return Null::Singleton();
}

Value library(State& state, List const& args) {
	checkNumArgs(args, 1);

	Character from = As<Character>(state, force(state, args[0]));
	if(from.length > 0) {
		loadLibrary(state, state.SymToStr(from[0]));
	}
	return Null::Singleton();
}

Value rm(State& state, List const& args) {
	for(int64_t i = 0; i < args.length; i++) 
		if(!expression(args[i]).isSymbol() && !expression(args[i]).isCharacter()) 
			_error("rm() arguments must be symbols or character vectors");
	for(int64_t i = 0; i < args.length; i++) {
		state.frame.environment->rm(Symbol(expression(args[i])));
	}
	return Null::Singleton();
}

Value sequence(State& state, List const& args) {
	checkNumArgs(args, 3);

	Value from = force(state, args[0]);
	Value by   = force(state, args[1]);
	Value len  = force(state, args[2]);

	double f = asReal1(from);
	double b = asReal1(by);
	double l = asReal1(len);

	return Sequence(f, b, l);	
}

Value repeat(State& state, List const& args) {
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

Value inherits(State& state, List const& args) {
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

Value attr(State& state, List const& args)
{
	checkNumArgs(args, 3);
	// NYI: exact
	Value object = force(state, args[0]);
	Character which = Character(force(state, args[1]));
	return getAttribute(object, which[0]);
}

Value assignAttr(State& state, List const& args)
{
	checkNumArgs(args, 3);
	Value object = force(state, args[0]);
	Character which = Character(force(state, args[1]));
	return setAttribute(object, which[0], force(state, args[2]));
}

Type::Enum cTypeCast(Value const& v, Type::Enum t)
{
	Type::Enum r;
	r = std::max(v.type, t);
	return r;
}

Value list(State& state, List const& args) {
	List out(args.length);
	for(int64_t i = 0; i < args.length; i++) out[i] = force(state, args[i]);
	out.attributes = args.attributes;
	return out;
}

Value unlist(State& state, List const& args) {
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
	Vector out = Vector(type, total);
	int64_t j = 0;
	for(int64_t i = 0; i < from.length; i++) {
		Insert(state, Vector(from[i]), 0, out, j, Vector(from[i]).length);
		j += from[i].length;
	}
	if(hasNames(from))
	{
		Character names = Character(getNames(from));
		Character outnames(total);
		int64_t j = 0;
		for(int64_t i = 0; i < (int64_t)from.length; i++) {
			for(int64_t m = 0; m < from[i].length; m++, j++) {
				// NYI: R makes these names distinct
				outnames[j] = names[i];
			}
		}
		setNames(out, outnames);
	}
	return out;
}

Vector Subset(State& state, Vector const& a, Vector const& i)	{
	if(Value(i).isDouble() || Value(i).isInteger()) {
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
				case Type::Null: return a; break;
				#define CASE(Name,...) case Type::Name: return SubsetInclude<Name>::eval(state, Name(a), index, positive); break;
				VECTOR_TYPES_NOT_NULL(CASE)
				#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};
		}
		else if(negative > 0) {
			switch(a.type) {
				case Type::Null: return a; break;
				#define CASE(Name) case Type::Name: return SubsetExclude<Name>::eval(state, Name(a), index, negative); break;
				VECTOR_TYPES_NOT_NULL(CASE)
				#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};	
		}
		else {
			return Vector(a.type, 0);
		}
	}
	else if(Value(i).isLogical()) {
		Logical index = Logical(i);
		switch(a.type) {
			case Type::Null: return a; break;
			#define CASE(Name) case Type::Name: return SubsetLogical<Name>::eval(state, Name(a), index); break;
			VECTOR_TYPES_NOT_NULL(CASE)
			#undef CASE
			default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
		};	
	}
	_error("NYI indexing type");
	return Null::Singleton();
}

Value subset(State& state, List const& args) {
	checkNumArgs(args, 2);
        Vector a = Vector(force(state, args[0]));
        Vector i = Vector(force(state, args[1]));
	return Subset(state, a,i);
}

Value subset2(State& state, List const& args) {
	checkNumArgs(args, 2);

        Value a = force(state, args[0]);
        Value b = force(state, args[1]);
	if(b.isCharacter() && hasNames(a)) {
		Symbol i = Character(b)[0];
		Character c = Character(getNames(a));
		
		int64_t j = 0;
		for(;j < c.length; j++) {
			if(c[j] == i)
				break;
		}
		if(j < c.length) {
			return Element2(Vector(a), j);
		}
	}
	else if(b.isInteger()) {
		return Element2(Vector(a), Integer(b)[0]-1);
	}
	else if(b.isDouble()) {
		return Element2(Vector(a), (int64_t)Double(b)[0]-1);
	}
	return Null::Singleton();
} 

Value dollar(State& state, List const& args) {
	checkNumArgs(args, 2);

        Value a = force(state, args[0]);
        int64_t i = Symbol(expression(args[1])).i;
	if(hasNames(a)) {
		Character c = Character(getNames(a));
		int64_t j = 0;
		for(;j < c.length; j++) {
			if(c[j] == i)
				break;
		}
		if(j < c.length) {
			return Element2(Vector(a), j);
		}
	}
	return Null::Singleton();
} 

Value length(State& state, List const& args) {
	checkNumArgs(args, 1);
	Vector a(force(state, args[0]));
	Integer i(1);
	i[0] = a.length;
	return i;
}

Value quote(State& state, List const& args) {
	checkNumArgs(args, 1);
	return expression(args[0]);
}

Value eval_fn(State& state, List const& args) {
	checkNumArgs(args, 3);
	Value expr = force(state, args[0]);
	Value envir = force(state, args[1]);
	//Value enclos = force(state, args[2]);
	return eval(state, Compiler::compile(state, expr), REnvironment(envir).ptr());
}

Value lapply(State& state, List const& args) {
	checkNumArgs(args, 2);
	List x = As<List>(state, force(state, args[0]));
	Value func = force(state, args[1]);

	Call apply(2);
	apply[0] = func;

	List result(x.length);
	// TODO: should have a way to make a simple function call without compiling,
	// or should have a fast case for compilation
	for(int64_t i = 0; i < x.length; i++) {
		apply[1] = x[i];
		result[i] = eval(state, Compiler::compile(state, apply));
	}

	return result;
}

Value tlist(State& state, List const& args) {
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
				element[j] = Element2(Vector(a[j]), i%a[j].length);
			else
				element[j] = a[j];
		}
		result[i] = element;
	}
	return result;
}

Value source(State& state, List const& args) {
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

Value switch_fn(State& state, List const& args) {
	Value one = force(state, args[0]);
	if(one.isInteger() && Integer(one).length == 1) {
		int64_t i = Integer(one)[0];
		if(i >= 1 && (int64_t)i <= args.length) {return force(state, args[i]);}
	} else if(one.isDouble() && Double(one).length == 1) {
		int64_t i = (int64_t)Double(one)[0];
		if(i >= 1 && (int64_t)i <= args.length) {return force(state, args[i]);}
	} else if(one.isCharacter() && Character(one).length == 1 && hasNames(args)) {
		Character names = Character(getNames(args));
		for(int64_t i = 1; i < args.length; i++) {
			if(names[i] == Character(one)[0]) {
				while(args[i].isNil() && i < args.length) i++;
				return i < args.length ? force(state, args[i]) : (Value)(Null::Singleton());
			}
		}
		for(int64_t i = 1; i < args.length; i++) {
			if(names[i] == Symbol::empty) {
				return force(state, args[i]);
			}
		}
	}
	return Null::Singleton();
}

Value environment(State& state, List const& args) {
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

Value parentframe(State& state, List const& args) {
	checkNumArgs(args, 1);
	int64_t i = (int64_t)asReal1(force(state, args[0]));
	Environment* env = state.frame.environment;
	while(i > 0 && env != env->DynamicParent()) {
		env = env->DynamicParent();
		i--;
	}
	return REnvironment(env);
}

Value stop_fn(State& state, List const& args) {
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

Value warning_fn(State& state, List const& args) {
	std::string message = "user warning";
	if(args.length > 0) {
		if(args[0].isCharacter() && Character(args[0]).length > 0) {
			message = state.SymToStr(Character(args[0])[0]);
		}
	}
	_warning(state, message);
	return Character::c(state.StrToSym(message));
} 

Value missing(State& state, List const& args) {
	Symbol s(expression(args[0])); 
	Value v =  state.frame.environment->get(s);
	return (v.isNil() || (v.isPromise() && Function(v).environment() == state.frame.environment)) ? Logical::True() : Logical::False();
}

Value max_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<FoldLeft, MaxOp>(state, a, result);
	return result;
}

Value min_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<FoldLeft, MinOp>(state, a, result);
	return result;
}

Value sum_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<FoldLeft, SumOp>(state, a, result);
	return result;
}

Value prod_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<FoldLeft, ProdOp>(state, a, result);
	return result;
}

Value cummax_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<ScanLeft, MaxOp>(state, a, result);
	return result;
}

Value cummin_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<ScanLeft, MinOp>(state, a, result);
	return result;
}

Value cumsum_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<ScanLeft, SumOp>(state, a, result);
	return result;
}

Value cumprod_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryArith<ScanLeft, ProdOp>(state, a, result);
	return result;
}

Value any_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryLogical<FoldLeft, AnyOp>(state, a, result);
	return result;
}

Value all_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryLogical<FoldLeft, AllOp>(state, a, result);
	return result;
}

Value isna_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryFilter<Zip1, IsNAOp>(state, a, result);
	return result;
}

Value isnan_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryFilter<Zip1, IsNaNOp>(state, a, result);
	return result;
}

Value nchar_fn(State& state, List const& args) {
	checkNumArgs(args, 3);
	Value a = force(state, args[0]);
	// NYI: type or allowNA
	Value result;
	unaryCharacter<Zip1, NcharOp>(state, a, result);
	return result;
}

Value nzchar_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryCharacter<Zip1, NzcharOp>(state, a, result);
	return result;
}

Value isfinite_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryFilter<Zip1, IsFiniteOp>(state, a, result);
	return result;
}

Value isinfinite_fn(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value a = force(state, args[0]);
	Value result;
	unaryFilter<Zip1, IsInfiniteOp>(state, a, result);
	return result;
}

Value paste(State& state, List const& args) {
	Character a = As<Character>(state, force(state, args[0]));
	Character sep = As<Character>(state, force(state, args[1]));
	std::string result = "";
	for(int64_t i = 0; i+1 < a.length; i++) {
		result = result + state.SymToStr(a[i]) + state.SymToStr(sep[0]);
	}
	if(a.length > 0) result = result + state.SymToStr(a[a.length-1]);
	return Character::c(state.StrToSym(result));
}

Value deparse(State& state, List const& args) {
	Value v = force(state, args[0]);
	return Character::c(state.StrToSym(state.deparse(v)));
}

Value substitute(State& state, List const& args) {
	checkNumArgs(args, 1);
	Value v = args[0];
	while(v.isPromise()) v = Function(v).code()->expression;
	
	if(v.isSymbol()) {
		Value r = state.frame.environment->get(Symbol(v));
		if(!r.isNil()) v = r;
		while(v.isPromise()) v = Function(v).code()->expression;
	}
 	return v;
}

Value type_of(State& state, List const& args) {
	// Should have a direct mapping from type to symbol.
	Value v = force(state, args[0]);
	return Character::c(state.StrToSym(Type::toString(v.type)));
}

Value get(State& state, List const& args) {
	checkNumArgs(args, 2);
	Character c = As<Character>(state, force(state, args[0]));
	REnvironment e(force(state, args[1]));
	Value v = e.ptr()->get(c[0]);
	if(v.isNil())
		return Null::Singleton();
	else
		return v;
}

Value exists(State& state, List const& args) {
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

Value systemtime(State& state, List const& args) {
	checkNumArgs(args, 1);
	uint64_t s = readTime();
	force(state, args[0]);
	uint64_t e = readTime();
	return Double::c((e-s)/(1000000.0));
}

void importCoreLibrary(State& state, Environment* env)
{
	env->assign(state.StrToSym("max"), CFunction(max_fn));
	env->assign(state.StrToSym("min"), CFunction(min_fn));
	env->assign(state.StrToSym("sum"), CFunction(sum_fn));
	env->assign(state.StrToSym("prod"), CFunction(prod_fn));
	env->assign(state.StrToSym("cummax"), CFunction(cummax_fn));
	env->assign(state.StrToSym("cummin"), CFunction(cummin_fn));
	env->assign(state.StrToSym("cumsum"), CFunction(cumsum_fn));
	env->assign(state.StrToSym("cumprod"), CFunction(cumprod_fn));
	env->assign(state.StrToSym("any"), CFunction(any_fn));
	env->assign(state.StrToSym("all"), CFunction(all_fn));
	env->assign(state.StrToSym("nchar"), CFunction(nchar_fn));
	env->assign(state.StrToSym("nzchar"), CFunction(nzchar_fn));
	env->assign(state.StrToSym("is.na"), CFunction(isna_fn));
	env->assign(state.StrToSym("is.nan"), CFunction(isnan_fn));
	env->assign(state.StrToSym("is.finite"), CFunction(isfinite_fn));
	env->assign(state.StrToSym("is.infinite"), CFunction(isinfinite_fn));
	
	env->assign(state.StrToSym("cat"), CFunction(cat));
	env->assign(state.StrToSym("library"), CFunction(library));
	env->assign(state.StrToSym("rm"), CFunction(rm));
	env->assign(state.StrToSym("inherits"), CFunction(inherits));
	
	env->assign(state.StrToSym("seq"), CFunction(sequence));
	env->assign(state.StrToSym("rep"), CFunction(repeat));
	
	env->assign(state.StrToSym("attr"), CFunction(attr));
	env->assign(state.StrToSym("attr<-"), CFunction(assignAttr));
	
	env->assign(state.StrToSym("list"), CFunction(list));
	env->assign(state.StrToSym("unlist"), CFunction(unlist));
	env->assign(state.StrToSym("length"), CFunction(length));
	
	env->assign(state.StrToSym("["), CFunction(subset));
	env->assign(state.StrToSym("[["), CFunction(subset2));
	env->assign(state.StrToSym("$"), CFunction(dollar));

	env->assign(state.StrToSym("switch"), CFunction(switch_fn));

	env->assign(state.StrToSym("eval"), CFunction(eval_fn));
	env->assign(state.StrToSym("quote"), CFunction(quote));
	env->assign(state.StrToSym("source"), CFunction(source));

	env->assign(state.StrToSym("lapply"), CFunction(lapply));
	env->assign(state.StrToSym("t.list"), CFunction(tlist));

	env->assign(state.StrToSym("environment"), CFunction(environment));
	env->assign(state.StrToSym("parent.frame"), CFunction(parentframe));
	env->assign(state.StrToSym("missing"), CFunction(missing));
	
	env->assign(state.StrToSym("stop"), CFunction(stop_fn));
	env->assign(state.StrToSym("warning"), CFunction(warning_fn));
	
	env->assign(state.StrToSym("paste"), CFunction(paste));
	env->assign(state.StrToSym("deparse"), CFunction(deparse));
	env->assign(state.StrToSym("substitute"), CFunction(substitute));
	
	env->assign(state.StrToSym("typeof"), CFunction(type_of));
	
	env->assign(state.StrToSym("get"), CFunction(get));
	env->assign(state.StrToSym("exists"), CFunction(exists));

	env->assign(state.StrToSym("system.time"), CFunction(systemtime));
}

