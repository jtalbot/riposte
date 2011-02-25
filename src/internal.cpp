
#include "internal.h"
#include <assert.h>
#include <math.h>

typedef unsigned char boolean;

Value force(State& state, Value const& v) { 
	if(v.type() == Type::I_promise) {
		eval(state, Block(v), state.env); 
		return state.stack->pop();
	} else if(v.type() == Type::I_sympromise) {
		Value value;
		state.env->get(state, Symbol(Block(v).code()[0].a), value);
		return value;
	} else return v; 
}
Value quoted(Value const& v) { 
	if(v.type() == Type::I_promise || v.type() == Type::I_sympromise) 
		return Block(v).expression();
	 else return v; 
}
Value code(Value const& v) {
	return v; 
}

// Casting functions
void cast(boolean i, double& o) {
	o = i==0 ? 0.0 : 1.0;
}
void cast(int64_t i, double& o) {
	o = (double)i;
}
void cast(boolean i, int64_t& o) {
	o = i==0 ? 0 : 1;
}
void cast(double i, boolean& o) {
	o = i==0 ? 0 : 1;
}
void cast(int64_t i, boolean& o) {
	o = i==0 ? 0 : 1;
}
void cast(boolean i, boolean& o) {
	o = i;
}
void cast(double i, double& o) {
	o = i;
}
void cast(int64_t i, int64_t& o) {
	o = i;
}

typedef void (*UnaryOp)(void* a, uint64_t ia, void* r, uint64_t ir);
typedef void (*BinaryOp)(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir);

template<typename AType, typename TAType, typename RType>
struct PosOp {
	static void eval(void* a, uint64_t ia, void* r, uint64_t ir) {
		TAType ta; cast(((AType*)a)[ia], ta);
		((RType*)r)[ir] = ta;
	}
};

template<typename AType, typename TAType, typename RType>
struct NegOp {
	static void eval(void* a, uint64_t ia, void* r, uint64_t ir) {
		TAType ta; cast(((AType*)a)[ia], ta);
		((RType*)r)[ir] = -ta;
	}
};

template<typename AType, typename TAType, typename RType>
struct LNegOp {
	static void eval(void* a, uint64_t ia, void* r, uint64_t ir) {
		TAType ta; cast(((AType*)a)[ia], ta);
		((RType*)r)[ir] = !ta;
	}
};


template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct AddOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta + tb;
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct SubOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta - tb;
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct MulOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta * tb; 
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct DivOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta / tb; 
	}
};

double Floor(double a, double b) { return floor(a/b); /* TODO: Replace with ugly R version */ }
int64_t Floor(int64_t a, int64_t b) { return a/b; }

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct IDivOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = Floor(ta, tb);
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct PowOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = pow(ta, tb);
	}
};

double Mod(double a, double b) { return a - Floor(a,b) * b; /* TODO: Replace with ugly R version */ }
int64_t Mod(int64_t a, int64_t b) { return a % b; }

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct ModOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = Mod(ta, tb);
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct LTOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta < tb;
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct GTOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta > tb;
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct EqOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta == tb;
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct NEqOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta != tb;
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct GEOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta >= tb;
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct LEOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta <= tb;
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct AndOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta && tb;
	}
};

template<typename AType, typename TAType, typename BType, typename TBType, typename RType>
struct OrOp {
	static void eval(void* a, uint64_t ia, void* b, uint64_t ib, void* r, uint64_t ir) {
		TAType ta, tb; cast(((AType*)a)[ia], ta); cast(((BType*)b)[ib], tb);
		((RType*)r)[ir] = ta || tb;
	}
};

template<UnaryOp func>
struct Zip1 {
	static void eval(Vector const& a, Type type, Vector& r)
	{
		if(a.length() == 1) {
			r = Vector(type, 1);
			func(a.data(), 0, r.data(), 0);
		}
		else if(a.length() == 0) {
			r = Vector(type, 0);
		}
		else {
			r = Vector(type, a.length());
			for(uint64_t i = 0; i < a.length(); ++i) {
				func(a.data(), i, r.data(), i);
			}
		}
	}
};

template<BinaryOp func>
struct Zip2 {
	static void eval(Vector const& a, Vector const& b, Type type, Vector& r)
	{
		if(a.length() == 1 && b.length() == 1) {
			r = Vector(type, 1);
			func(a.data(), 0, b.data(), 0, r.data(), 0);
		}
		else if(a.length() == 0 || b.length() == 0) {
			r = Vector(type, 0);
		}
		else if(b.length() == 1) {
			r = Vector(type, a.length());
			for(uint64_t i = 0; i < a.length(); ++i) {
				func(a.data(), i, b.data(), 0, r.data(), i);
			}
		}
		else if(a.length() == 1) {
			r = Vector(type, b.length());
			for(uint64_t i = 0; i < b.length(); ++i) {
				func(a.data(), 0, b.data(), i, r.data(), i);
			}
		}
		else if(a.length() >= b.length()) {
			r = Vector(type, a.length());
			uint64_t j = 0;
			for(uint64_t i = 0; i < a.length(); ++i) {
				func(a.data(), i, b.data(), j, r.data(), i);
				++j;
				if(j >= b.length())
					j = 0;
			}
		}
		else {
			r = Vector(type, b.length());
			uint64_t j = 0;
			for(uint64_t i = 0; i < b.length(); ++i) {
				func(a.data(), j, b.data(), i, r.data(), i);
				++j;
				if(j >= a.length())
					j = 0;
			}
		}
	}
};

template< 
	template<UnaryOp func> class VectorOp,
	template<typename AType, typename TAType, typename RType> class Op > 
uint64_t unaryArith(State& state, uint64_t nargs) {

	assert(nargs == 1);
	
	Stack& stack = *(state.stack);
	
	Value a = force(state, stack.pop());	

	Vector r;
	if(a.type() == Type::R_double) {
		VectorOp<Op<double, double, double>::eval>
			::eval(Vector(a), Type::R_double, r);
	}
	else if(a.type() == Type::R_integer) {
		VectorOp<Op<int64_t, int64_t, int64_t>::eval>
			::eval(Vector(a), Type::R_integer, r);
	}
	else if(a.type() == Type::R_logical) {
		VectorOp<Op<boolean, int64_t, int64_t>::eval>
			::eval(Vector(a), Type::R_integer, r);
	}
	else {
		printf("non-numeric argument to unary numeric operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
};

template< 
	template<UnaryOp func> class VectorOp,
	template<typename AType, typename TAType, typename RType> class Op > 
uint64_t unaryLogical(State& state, uint64_t nargs) {

	assert(nargs == 1);
	
	Stack& stack = *(state.stack);
	
	Value a = force(state, stack.pop());	

	Vector r;
	if(a.type() == Type::R_double) {
		VectorOp<Op<double, boolean, boolean>::eval>
			::eval(Vector(a), Type::R_logical, r);
	}
	else if(a.type() == Type::R_integer) {
		VectorOp<Op<int64_t, boolean, boolean>::eval>
			::eval(Vector(a), Type::R_logical, r);
	}
	else if(a.type() == Type::R_logical) {
		VectorOp<Op<boolean, boolean, boolean>::eval>
			::eval(Vector(a), Type::R_logical, r);
	}
	else {
		printf("non-numeric argument to unary logical operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
};


template< 
template<BinaryOp func> class VectorOp,
template<typename AType, typename TAType, typename BType, typename TBType, typename RType> class Op > 
uint64_t binaryArith(State& state, uint64_t nargs) {

	assert(nargs == 2);
	
	Stack& stack = *(state.stack);
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type() == Type::R_double && b.type() == Type::R_double) {
		VectorOp<Op<double, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_integer && b.type() == Type::R_double) {
		VectorOp<Op<int64_t, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_double && b.type() == Type::R_integer) {
		VectorOp<Op<double, double, int64_t, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_integer && b.type() == Type::R_integer) {
		VectorOp<Op<int64_t, int64_t, int64_t, int64_t, int64_t>::eval>
			::eval(Vector(a), Vector(b), Type::R_integer, r);
	}
	else if(a.type() == Type::R_double && b.type() == Type::R_logical) {
		VectorOp<Op<double, double, boolean, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_logical && b.type() == Type::R_double) {
		VectorOp<Op<boolean, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_integer && b.type() == Type::R_logical) {
		VectorOp<Op<int64_t, int64_t, boolean, int64_t, int64_t>::eval>
			::eval(Vector(a), Vector(b), Type::R_integer, r);
	}
	else if(a.type() == Type::R_logical && b.type() == Type::R_integer) {
		VectorOp<Op<boolean, int64_t, int64_t, int64_t, int64_t>::eval>
			::eval(Vector(a), Vector(b), Type::R_integer, r);
	}
	else if(a.type() == Type::R_logical && b.type() == Type::R_logical) {
		VectorOp<Op<boolean, int64_t, boolean, int64_t, int64_t>::eval>
			::eval(Vector(a), Vector(b), Type::R_integer, r);
	}
	else {
		printf("non-numeric argument to binary numeric operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
}

template< 
template<BinaryOp func> class VectorOp,
template<typename AType, typename TAType, typename BType, typename TBType, typename RType> class Op > 
uint64_t binaryDoubleArith(State& state, uint64_t nargs) {

	assert(nargs == 2);
	
	Stack& stack = *(state.stack);
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type() == Type::R_double && b.type() == Type::R_double) {
		VectorOp<Op<double, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_integer && b.type() == Type::R_double) {
		VectorOp<Op<int64_t, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_double && b.type() == Type::R_integer) {
		VectorOp<Op<double, double, int64_t, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_integer && b.type() == Type::R_integer) {
		VectorOp<Op<int64_t, double, int64_t, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_double && b.type() == Type::R_logical) {
		VectorOp<Op<double, double, boolean, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_logical && b.type() == Type::R_double) {
		VectorOp<Op<boolean, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_integer && b.type() == Type::R_logical) {
		VectorOp<Op<int64_t, double, boolean, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_logical && b.type() == Type::R_integer) {
		VectorOp<Op<boolean, double, int64_t, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type() == Type::R_logical && b.type() == Type::R_logical) {
		VectorOp<Op<boolean, double, boolean, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else {
		printf("non-numeric argument to numeric operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
}

template< 
template<BinaryOp func> class VectorOp,
template<typename AType, typename TAType, typename BType, typename TBType, typename RType> class Op > 
uint64_t binaryLogical(State& state, uint64_t nargs) {

	assert(nargs == 2);
	
	Stack& stack = *(state.stack);
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type() == Type::R_double && b.type() == Type::R_double) {
		VectorOp<Op<double, boolean, double, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type() == Type::R_integer && b.type() == Type::R_double) {
		VectorOp<Op<int64_t, boolean, double, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type() == Type::R_double && b.type() == Type::R_integer) {
		VectorOp<Op<double, boolean, int64_t, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type() == Type::R_integer && b.type() == Type::R_integer) {
		VectorOp<Op<int64_t, boolean, int64_t, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type() == Type::R_double && b.type() == Type::R_logical) {
		VectorOp<Op<double, boolean, boolean, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type() == Type::R_logical && b.type() == Type::R_double) {
		VectorOp<Op<boolean, boolean, double, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type() == Type::R_integer && b.type() == Type::R_logical) {
		VectorOp<Op<int64_t, boolean, boolean, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type() == Type::R_logical && b.type() == Type::R_integer) {
		VectorOp<Op<boolean, boolean, int64_t, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type() == Type::R_logical && b.type() == Type::R_logical) {
		VectorOp<Op<boolean, boolean, boolean, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else {
		printf("non-logical argument to logical operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
}

/*uint64_t add(State& state, uint64_t nargs)
{
	Stack& stack = *(state.stack);
	
	Value arg0 = force(state, stack.pop());	
	Value arg1 = force(state, stack.pop());	
	Value& result = stack.reserve();

	zip2<AddOp>
	Value::setDouble(result, arg0.getDouble()+arg1.getDouble());	
	return 1;
}*/

/*
uint64_t zip2(State& state, uint64_t nargs)
{
	Stack& stack = *(state.stack);
	
	//printf("zip: %d\n", stack.top);
	
	if(nargs != 3) {
		printf("Zip2 needs three parameters\n");
		Value::set(stack.reserve(), Type::R_null, 0);
	}

	Value arg0 = force(state, stack.pop());	
	Value arg1 = force(state, stack.pop());	
	Value arg2 = force(state, stack.pop());	

	BinaryOp func = (BinaryOp)(arg2.ptr());

	Value& result = stack.reserve();
	
	if(arg0.type() == Type::R_scalardouble && 
		arg1.type() == Type::R_scalardouble) {
		func(arg0, arg1, result);
	}
	else if(arg1.type() == Type::R_scalardouble) {
		Vector r0(arg0);
		uint64_t n1 = r0.length();	
		uint64_t n2 = 1;	
		uint64_t r = n1 == 0 || n2 == 0 ? 0 : n1 > n2 ? n1 : n2;
		if(r == 1) func(r0.get(0), arg1, result);
		else {
			Vector p;
			Real real(r);
			real.toVector(p);
			Value v;
			for(uint64_t i = 0; i < r; i++) {
				func(r0.get(i), arg1, v);
				p.set(i, v);
			}
			p.toValue(result);
		}
	}
	else if(arg0.type() == Type::R_scalardouble) {
		Vector r1(arg1);
		uint64_t n1 = 1;	
		uint64_t n2 = r1.length();	
		uint64_t r = n1 == 0 || n2 == 0 ? 0 : n1 > n2 ? n1 : n2;
		if(r == 1) func(arg0, r1.get(0), result);
		else {
			Vector p;
			Real real(r);
			real.toVector(p);
			Value v;
			for(uint64_t i = 0; i < r; i++) {
				func(arg0, r1.get(i), v);
				p.set(i, v);
			}
			p.toValue(result);
		}
	}
	else {
		Vector r0(arg0);
		Vector r1(arg1);

		uint64_t n1 = r0.length();	
		uint64_t n2 = r1.length();	

		uint64_t r = n1 == 0 || n2 == 0 ? 0 : n1 > n2 ? n1 : n2;
		if(r == 1) func(r0.get(0), r1.get(1), result);
		else {
			Vector p;
			Real real(r);
			real.toVector(p);
			Value v;
			for(uint64_t i = 0; i < r; i++) {
				func(r0.get(i), r1.get(i), v);
				p.set(i, v);		
			}
			p.toValue(result);
		}
	}
	return 1;
}
*/
uint64_t assign(State& state, uint64_t nargs) {
	return 1;
}

uint64_t forloop(State& state, uint64_t nargs) {
	return 1;
}

uint64_t function(State& state, uint64_t nargs)
{
	assert(nargs == 3);
	
	Stack& stack = *(state.stack);
	
	Value arg0 = force(state, stack.pop());
	Value arg1 = code(stack.pop());
	Value arg2 = force(state, stack.pop());

	/*Function::ParameterList formals;
	bool vararg = false;
	Function::ParameterList after;

	List arguments = List(arg0);
	Character argNames = Character(arguments.names());
	uint64_t length = arguments.length();
	for(uint64_t i = 0; i < length; i++) {
		std::string name = argNames[i];
		if(name == "...")
			vararg = true;
		else {
			Function::Parameter p;
			p.name = Symbol(name);
			p.def = arguments[i];
			if(!vararg)
				formals.push_back(p);
			else
				after.push_back(p);
		}
	}*/

	Function func(List(arg0), arg1, state.env);
	Value result;
	func.toValue(result);
	stack.push(result);

	return 1;
}

uint64_t rm(State& state, uint64_t nargs) {
	assert(nargs == 1);
	Stack& stack = *(state.stack);
	Value symbol = quoted(stack.pop());
	state.env->rm(Symbol(symbol));
	stack.push(Value::null);
	return 1;
}

uint64_t curlyBrackets(State& state, uint64_t nargs) {
	return 1;
}

uint64_t parentheses(State& state, uint64_t nargs) {
	return 1;
}

uint64_t sequence(State& state, uint64_t nargs)
{
	assert(nargs == 3);

	Stack& stack = *(state.stack);
	
	Value from = force(state, stack.pop());
	Value by   = force(state, stack.pop());
	Value len  = force(state, stack.pop());
	
	double f = asReal1(from);
	double b = asReal1(by);
	double l = asReal1(len);

	Double r(l);
	double j = 0;
	for(uint64_t i = 0; i < l; i++) {
		r[i] = f+j;
		j = j + b;
	}
	Value v;
	r.toValue(stack.reserve());
	
	return 1;
}

uint64_t repeat(State& state, uint64_t nargs)
{
	assert(nargs == 3);
	Stack& stack = *(state.stack);
	
	Value vec  = force(state, stack.pop());
	Value each = force(state, stack.pop());
	Value len  = force(state, stack.pop());
	
	double v = asReal1(vec);
	//double e = asReal1(each);
	double l = asReal1(len);
	
	Double r(l);
	for(uint64_t i = 0; i < l; i++) {
		r[i] = v;
	}
	r.toValue(stack.reserve());
	return 1;
}

uint64_t typeOf(State& state, uint64_t nargs)
{
	assert(nargs == 1);
	Stack& stack = *(state.stack);
	Character c(1);
	c[0] = force(state, stack.pop()).type().toString();
	c.toValue(stack.reserve());
	return 1;
}

uint64_t mode(State& state, uint64_t nargs)
{
	assert(nargs == 1);
	Stack& stack = *(state.stack);
	Character c(1);
	Value v = force(state, stack.pop());
	if(v.type() == Type::R_integer || v.type() == Type::R_double)
		c[0] = "numeric";
	else if(v.type() == Type::R_symbol)
		c[0] = "name";
	else
		c[0] = v.type().toString();
	c.toValue(stack.reserve());
	return 1;
}


uint64_t plusOp(State& state, uint64_t nargs) {
	if(nargs == 1)
		return unaryArith<Zip1, PosOp>(state, nargs);
	else
		return binaryArith<Zip2, AddOp>(state, nargs);
}

uint64_t minusOp(State& state, uint64_t nargs) {
	if(nargs == 1)
		return unaryArith<Zip1, NegOp>(state, nargs);
	else
		return binaryArith<Zip2, SubOp>(state, nargs);
}

void addMathOps(Environment* env)
{
	Value v;

	// operators that are implemented as byte codes, thus, no actual implemention is necessary here.
	CFunction(forloop).toValue(v);
	env->assign(Symbol("for"), v);
	CFunction(assign).toValue(v);
	env->assign(Symbol("<-"), v);
	CFunction(curlyBrackets).toValue(v);
	env->assign(Symbol("{"), v);
	CFunction(parentheses).toValue(v);
	env->assign(Symbol("("), v);

	CFunction(plusOp).toValue(v);
	env->assign(Symbol("+"), v);
	CFunction(minusOp).toValue(v);
	env->assign(Symbol("-"), v);
	CFunction::Cffi mul = binaryArith<Zip2, MulOp>;
	CFunction(mul).toValue(v);
	env->assign(Symbol("*"), v);
	CFunction::Cffi div = binaryDoubleArith<Zip2, DivOp>;
	CFunction(div).toValue(v);
	env->assign(Symbol("/"), v);
	CFunction::Cffi idiv = binaryArith<Zip2, IDivOp>;
	CFunction(idiv).toValue(v);
	env->assign(Symbol("%/%"), v);
	CFunction::Cffi pow = binaryDoubleArith<Zip2, PowOp>;
	CFunction(pow).toValue(v);
	env->assign(Symbol("^"), v);
	CFunction::Cffi mod = binaryArith<Zip2, ModOp>;
	CFunction(mod).toValue(v);
	env->assign(Symbol("%%"), v);

	CFunction::Cffi lneg = unaryLogical<Zip1, LNegOp>;
	CFunction(lneg).toValue(v);
	env->assign(Symbol("!"), v);
	CFunction::Cffi And = binaryLogical<Zip2, AndOp>;
	CFunction(And).toValue(v);
	env->assign(Symbol("&"), v);
	CFunction::Cffi Or = binaryLogical<Zip2, OrOp>;
	CFunction(Or).toValue(v);
	env->assign(Symbol("|"), v);
	
	CFunction(function).toValue(v);
	env->assign(Symbol("function"), v);
	CFunction(rm).toValue(v);
	env->assign(Symbol("rm"), v);
	CFunction(typeOf).toValue(v);
	env->assign(Symbol("typeof"), v);
	env->assign(Symbol("storage.mode"), v);
	CFunction(mode).toValue(v);
	env->assign(Symbol("mode"), v);
	
	CFunction(sequence).toValue(v);
	env->assign(Symbol("seq"), v);
	CFunction(repeat).toValue(v);
	env->assign(Symbol("rep"), v);

	//Value::set(v, Type::R_op, (void*)addVPrimitive);
	//env->assign("+.vprimitive", v);
}


CFunction::Cffi AddInternal = plusOp;
