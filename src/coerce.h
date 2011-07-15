
#ifndef _RIPOSTE_COERCE_H
#define _RIPOSTE_COERCE_H

#include "value.h"
#include "vector.h"

#include <limits>

// Casting functions between types

template<typename I, typename O>
static typename O::Element Cast(State& state, typename I::Element const& i) { return (typename O::Element)i; }

//gcc >= 4.3 requires template specialization to have the same storage class (e.g. 'static') as the orignal template
//specifying static is an error
//gcc < 4.3 treats the declarations as distinct. Not specifying a storage class makes the specialization external >_<
#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
#define SPECIALIZED_STATIC
#else
#define SPECIALIZED_STATIC static
#endif

template<>
SPECIALIZED_STATIC Integer::Element Cast<Logical, Integer>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Integer::NAelement : i ? 1 : 0; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Logical, Double>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Double::NAelement : i ? 1.0 : 0.0; }

template<>
SPECIALIZED_STATIC Complex::Element Cast<Logical, Complex>(State& state, Logical::Element const& i) { if(Logical::isNA(i)) return Complex::NAelement; else if(i) return std::complex<double>(1.0,0.0); else return std::complex<double>(0.0,0.0); }

template<>
SPECIALIZED_STATIC Character::Element Cast<Logical, Character>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Character::NAelement : i ? Symbol::TRUE : Symbol::FALSE; }

template<>
SPECIALIZED_STATIC List::Element Cast<Logical, List>(State& state, Logical::Element const& i) { return Logical::c(i); }

template<>
SPECIALIZED_STATIC Call::Element Cast<Logical, Call>(State& state, Logical::Element const& i) { return Logical::c(i); }


template<>
SPECIALIZED_STATIC Logical::Element Cast<Integer, Logical>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Logical::NAelement : i != 0 ? 1 : 0; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Integer, Double>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Double::NAelement : (Double::Element)i; }

template<>
SPECIALIZED_STATIC Complex::Element Cast<Integer, Complex>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Complex::NAelement : std::complex<double>(i, 0); }

template<>
SPECIALIZED_STATIC Character::Element Cast<Integer, Character>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Character::NAelement : state.StrToSym(intToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Integer, List>(State& state, Integer::Element const& i) { return Integer::c(i); }

template<>
SPECIALIZED_STATIC Call::Element Cast<Integer, Call>(State& state, Integer::Element const& i) { return Integer::c(i); }


template<>
SPECIALIZED_STATIC Logical::Element Cast<Double, Logical>(State& state, Double::Element const& i) { return Double::isNA(i) ? Logical::NAelement : i != 0.0 ? 1 : 0; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Double, Integer>(State& state, Double::Element const& i) { return Double::isNA(i) || i > std::numeric_limits<Integer::Element>::max() || i < std::numeric_limits<Integer::Element>::min() ? Integer::NAelement : i;  }

template<>
SPECIALIZED_STATIC Complex::Element Cast<Double, Complex>(State& state, Double::Element const& i) { return Double::isNA(i) ? Complex::NAelement : std::complex<double>(i, 0);  }

template<>
SPECIALIZED_STATIC Character::Element Cast<Double, Character>(State& state, Double::Element const& i) { return Integer::isNA(i) ? Character::NAelement : state.StrToSym(doubleToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Double, List>(State& state, Double::Element const& i) { return Double::c(i); }

template<>
SPECIALIZED_STATIC Call::Element Cast<Double, Call>(State& state, Double::Element const& i) { return Double::c(i); }


template<>
SPECIALIZED_STATIC Logical::Element Cast<Complex, Logical>(State& state, Complex::Element const& i) { if(Complex::isNA(i)) return Logical::NAelement; else return i.real() != 0 || i.imag() != 0; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Complex, Integer>(State& state, Complex::Element const& i) { return (Complex::isNA(i) || i.real() > std::numeric_limits<Integer::Element>::max() || i.real() < std::numeric_limits<Integer::Element>::min() || i.imag() != 0) ? Integer::NAelement : i.real();  }

template<>
SPECIALIZED_STATIC Double::Element Cast<Complex, Double>(State& state, Complex::Element const& i) { if(Complex::isNA(i) || i.imag() != 0) return Double::NAelement; else return i.real(); }

template<>
SPECIALIZED_STATIC Character::Element Cast<Complex, Character>(State& state, Complex::Element const& i) { if(Complex::isNA(i)) return Character::NAelement; else return state.StrToSym(complexToStr(i));}

template<>
SPECIALIZED_STATIC List::Element Cast<Complex, List>(State& state, Complex::Element const& i) { return Complex::c(i); }

template<>
SPECIALIZED_STATIC Call::Element Cast<Complex, Call>(State& state, Complex::Element const& i) { return Complex::c(i); }


template<>
SPECIALIZED_STATIC Logical::Element Cast<Character, Logical>(State& state, Character::Element const& i) { if(i == Symbol::TRUE) return 1; else if(i == Symbol::FALSE) return 0; else return Logical::NAelement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Character, Integer>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Integer::NAelement; else {try{return strToInt(state.SymToStr(i));} catch(...) {return Integer::NAelement;}} }

template<>
SPECIALIZED_STATIC Double::Element Cast<Character, Double>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Double::NAelement; else {try{return strToDouble(state.SymToStr(i));} catch(...) {return Double::NAelement;}} }

template<>
SPECIALIZED_STATIC Complex::Element Cast<Character, Complex>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Complex::NAelement; else {try{return strToComplex(state.SymToStr(i));} catch(...) {return Complex::NAelement;}} }

template<>
SPECIALIZED_STATIC List::Element Cast<Character, List>(State& state, Character::Element const& i) { return Character::c(i); }

template<>
SPECIALIZED_STATIC Call::Element Cast<Character, Call>(State& state, Character::Element const& i) { return Character::c(i); }


template<class I, class O> 
struct CastOp : public UnaryOp<I, O> {
	static typename CastOp::R eval(State& state, typename CastOp::A const& i) { return Cast<typename CastOp::AV, typename CastOp::RV>(state, i); }
};


template<class O>
O As(State& state, Value const& src) {
	if(src.type == O::type)
		return src;
	switch(src.type.Enum()) {
		case Type::E_R_null: return O(0); break;
		case Type::E_R_double: return Zip1< CastOp<Double, O> >::eval(state, src); break;
		case Type::E_R_integer: return Zip1< CastOp<Integer, O> >::eval(state, src); break;
		case Type::E_R_logical: return Zip1< CastOp<Logical, O> >::eval(state, src); break;
		case Type::E_R_complex: return Zip1< CastOp<Complex, O> >::eval(state, src); break;
		case Type::E_R_character: return Zip1< CastOp<Character, O> >::eval(state, src); break;
		case Type::E_R_list: return Zip1< CastOp<List, O> >::eval(state, src); break;
		case Type::E_R_call: return Zip1< CastOp<Call, O> >::eval(state, src); break;
		default: _error(std::string("Invalid cast from ") + src.type.toString() + " to " + O::type.toString()); break;
	};
}
template<>
inline Null As<Null>(State& state, Value const& src) {
	return Null::singleton;
}

inline Value As(State& state, Type type, Value const& src) {
	if(src.type == type)
		return src;
	switch(type.Enum()) {
		case Type::E_R_null: return As<Null>(state, src); break;
		case Type::E_R_double: return As<Double>(state, src); break;
		case Type::E_R_integer: return As<Integer>(state, src); break;
		case Type::E_R_logical: return As<Logical>(state, src); break;
		case Type::E_R_complex: return As<Complex>(state, src); break;
		case Type::E_R_character: return As<Character>(state, src); break;
		case Type::E_R_list: return As<List>(state, src); break;
		case Type::E_R_call: return As<Call>(state, src); break;
		default: _error(std::string("Invalid cast from ") + src.type.toString() + " to " + type.toString()); break;
	};
}


// Use As, so have to be after...

template<>
SPECIALIZED_STATIC Logical::Element Cast<List, Logical>(State& state, List::Element const& i) { Logical a = As<Logical>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC Integer::Element Cast<List, Integer>(State& state, List::Element const& i) { Integer a = As<Integer>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC Double::Element Cast<List, Double>(State& state, List::Element const& i) { Double a = As<Double>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC Complex::Element Cast<List, Complex>(State& state, List::Element const& i) { Complex a = As<Complex>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC Character::Element Cast<List, Character>(State& state, List::Element const& i) { Character a = As<Character>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC Call::Element Cast<List, Call>(State& state, Call::Element const& i) { return i; }


template<>
SPECIALIZED_STATIC Logical::Element Cast<Call, Logical>(State& state, Call::Element const& i) { Logical a = As<Logical>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Call, Integer>(State& state, Call::Element const& i) { Integer a = As<Integer>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC Double::Element Cast<Call, Double>(State& state, Call::Element const& i) { Double a = As<Double>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC Complex::Element Cast<Call, Complex>(State& state, Call::Element const& i) { Complex a = As<Complex>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC Character::Element Cast<Call, Character>(State& state, Call::Element const& i) { Character a = As<Character>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast"); }

template<>
SPECIALIZED_STATIC List::Element Cast<Call, List>(State& state, Call::Element const& i) { return i; }


void importCoerceFunctions(State& state, Environment* env);

#endif
