
#ifndef _RIPOSTE_INTERNAL_H
#define _RIPOSTE_INTERNAL_H

#include "value.h"
#include <assert.h>
#include <math.h>

void addMathOps(State& state);


typedef unsigned char boolean;

inline Value force(State& state, Value const& v) { 
	if(v.type == Type::I_promise) {
		eval(state, Block(v), state.env); 
		return state.stack.pop();
	} else if(v.type == Type::I_sympromise) {
		Value value;
		state.env->get(state, v.i, value);
		return value;
	} else return v; 
}
inline Value quoted(Value const& v) { 
	if(v.type == Type::I_promise || v.type == Type::I_sympromise) 
		return Block(v).expression();
	 else return v; 
}
inline Value code(Value const& v) {
	return v; 
}

// Casting functions
inline void cast(boolean i, double& o) {
	o = i==0 ? 0.0 : 1.0;
}
inline void cast(int64_t i, double& o) {
	o = (double)i;
}
inline void cast(boolean i, int64_t& o) {
	o = i==0 ? 0 : 1;
}
inline void cast(double i, boolean& o) {
	o = i==0 ? 0 : 1;
}
inline void cast(int64_t i, boolean& o) {
	o = i==0 ? 0 : 1;
}
inline void cast(boolean i, boolean& o) {
	o = i;
}
inline void cast(double i, double& o) {
	o = i;
}
inline void cast(int64_t i, int64_t& o) {
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

inline double Floor(double a, double b) { return floor(a/b); /* TODO: Replace with ugly R version */ }
inline int64_t Floor(int64_t a, int64_t b) { return a/b; }

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

inline double Mod(double a, double b) { return a - Floor(a,b) * b; /* TODO: Replace with ugly R version */ }
inline int64_t Mod(int64_t a, int64_t b) { return a % b; }

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
struct NeqOp {
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
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double) {
		VectorOp<Op<double, double, double>::eval>
			::eval(Vector(a), Type::R_double, r);
	}
	else if(a.type == Type::R_integer) {
		VectorOp<Op<int64_t, int64_t, int64_t>::eval>
			::eval(Vector(a), Type::R_integer, r);
	}
	else if(a.type == Type::R_logical) {
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
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double) {
		VectorOp<Op<double, boolean, boolean>::eval>
			::eval(Vector(a), Type::R_logical, r);
	}
	else if(a.type == Type::R_integer) {
		VectorOp<Op<int64_t, boolean, boolean>::eval>
			::eval(Vector(a), Type::R_logical, r);
	}
	else if(a.type == Type::R_logical) {
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
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double && b.type == Type::R_double) {
		VectorOp<Op<double, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_double) {
		VectorOp<Op<int64_t, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_integer) {
		VectorOp<Op<double, double, int64_t, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_integer) {
		VectorOp<Op<int64_t, int64_t, int64_t, int64_t, int64_t>::eval>
			::eval(Vector(a), Vector(b), Type::R_integer, r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_logical) {
		VectorOp<Op<double, double, boolean, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_double) {
		VectorOp<Op<boolean, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_logical) {
		VectorOp<Op<int64_t, int64_t, boolean, int64_t, int64_t>::eval>
			::eval(Vector(a), Vector(b), Type::R_integer, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_integer) {
		VectorOp<Op<boolean, int64_t, int64_t, int64_t, int64_t>::eval>
			::eval(Vector(a), Vector(b), Type::R_integer, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_logical) {
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
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double && b.type == Type::R_double) {
		VectorOp<Op<double, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_double) {
		VectorOp<Op<int64_t, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_integer) {
		VectorOp<Op<double, double, int64_t, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_integer) {
		VectorOp<Op<int64_t, double, int64_t, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_logical) {
		VectorOp<Op<double, double, boolean, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_double) {
		VectorOp<Op<boolean, double, double, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_logical) {
		VectorOp<Op<int64_t, double, boolean, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_integer) {
		VectorOp<Op<boolean, double, int64_t, double, double>::eval>
			::eval(Vector(a), Vector(b), Type::R_double, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_logical) {
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
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double && b.type == Type::R_double) {
		VectorOp<Op<double, boolean, double, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_double) {
		VectorOp<Op<int64_t, boolean, double, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_integer) {
		VectorOp<Op<double, boolean, int64_t, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_integer) {
		VectorOp<Op<int64_t, boolean, int64_t, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_logical) {
		VectorOp<Op<double, boolean, boolean, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_double) {
		VectorOp<Op<boolean, boolean, double, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_logical) {
		VectorOp<Op<int64_t, boolean, boolean, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_integer) {
		VectorOp<Op<boolean, boolean, int64_t, boolean, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_logical) {
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

template< 
template<BinaryOp func> class VectorOp,
template<typename AType, typename TAType, typename BType, typename TBType, typename RType> class Op > 
uint64_t binaryOrdinal(State& state, uint64_t nargs) {

	assert(nargs == 2);
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double && b.type == Type::R_double) {
		VectorOp<Op<double, double, double, double, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_double) {
		VectorOp<Op<int64_t, double, double, double, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_integer) {
		VectorOp<Op<double, double, int64_t, double, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_integer) {
		VectorOp<Op<int64_t, int64_t, int64_t, int64_t, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_logical) {
		VectorOp<Op<double, double, boolean, double, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_double) {
		VectorOp<Op<boolean, double, double, double, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_logical) {
		VectorOp<Op<int64_t, int64_t, boolean, int64_t, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_integer) {
		VectorOp<Op<boolean, int64_t, int64_t, int64_t, boolean>::eval>
			::eval(Vector(a), Vector(b), Type::R_logical, r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_logical) {
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


#endif

