
#ifndef _RIPOSTE_COERCE_H
#define _RIPOSTE_COERCE_H

#include "value.h"
#include "vector.h"

#include <limits>

// Casting functions between types

template<typename I, typename O>
static typename O::Element Cast(State& state, typename I::Element const& i) { return (typename O::Element)i; }

template<>
static Integer::Element Cast<Logical, Integer>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Integer::NAelement : i ? 1 : 0; }

template<>
static Double::Element Cast<Logical, Double>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Double::NAelement : i ? 1.0 : 0.0; }

template<>
static Complex::Element Cast<Logical, Complex>(State& state, Logical::Element const& i) { if(Logical::isNA(i)) return Complex::NAelement; else if(i) return std::complex<double>(1.0,0.0); else return std::complex<double>(0.0,0.0); }

template<>
static Character::Element Cast<Logical, Character>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Character::NAelement : i ? Symbol::TRUE : Symbol::FALSE; }

template<>
static List::Element Cast<Logical, List>(State& state, Logical::Element const& i) { return Logical::c(i); }


template<>
static Logical::Element Cast<Integer, Logical>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Logical::NAelement : i != 0 ? 1 : 0; }

template<>
static Double::Element Cast<Integer, Double>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Double::NAelement : i != 0 ? 1.0 : 0.0; }

template<>
static Complex::Element Cast<Integer, Complex>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Complex::NAelement : std::complex<double>(i, 0); }

template<>
static Character::Element Cast<Integer, Character>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Character::NAelement : Symbol(state.inString(intToStr(i))); }

template<>
static List::Element Cast<Integer, List>(State& state, Integer::Element const& i) { return Integer::c(i); }


template<>
static Logical::Element Cast<Double, Logical>(State& state, Double::Element const& i) { return Double::isNA(i) ? Logical::NAelement : i != 0.0 ? 1 : 0; }

template<>
static Integer::Element Cast<Double, Integer>(State& state, Double::Element const& i) { return Double::isNA(i) || i > std::numeric_limits<Integer::Element>::max() || i < std::numeric_limits<Integer::Element>::min() ? Integer::NAelement : i;  }

template<>
static Complex::Element Cast<Double, Complex>(State& state, Double::Element const& i) { return Double::isNA(i) ? Complex::NAelement : std::complex<double>(i, 0);  }

template<>
static Character::Element Cast<Double, Character>(State& state, Double::Element const& i) { return Integer::isNA(i) ? Character::NAelement : Symbol(state.inString(doubleToStr(i))); }

template<>
static List::Element Cast<Double, List>(State& state, Double::Element const& i) { return Double::c(i); }


template<>
static Logical::Element Cast<Complex, Logical>(State& state, Complex::Element const& i) { if(Complex::isNA(i)) return Logical::NAelement; else return i.real() != 0 || i.imag() != 0; }

template<>
static Integer::Element Cast<Complex, Integer>(State& state, Complex::Element const& i) { return (Complex::isNA(i) || i.real() > std::numeric_limits<Integer::Element>::max() || i.real() < std::numeric_limits<Integer::Element>::min() || i.imag() != 0) ? Integer::NAelement : i.real();  }

template<>
static Double::Element Cast<Complex, Double>(State& state, Complex::Element const& i) { if(Complex::isNA(i) || i.imag() != 0) return Double::NAelement; else return i.real(); }

template<>
static Character::Element Cast<Complex, Character>(State& state, Complex::Element const& i) { if(Complex::isNA(i)) return Character::NAelement; else return Symbol(state.inString(complexToStr(i)));}

template<>
static List::Element Cast<Complex, List>(State& state, Complex::Element const& i) { return Complex::c(i); }


template<>
static Logical::Element Cast<Character, Logical>(State& state, Character::Element const& i) { if(i == Symbol::TRUE) return 1; else if(i == Symbol::FALSE) return 0; else return Logical::NAelement; }

template<>
static Integer::Element Cast<Character, Integer>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Integer::NAelement; else {try{return strToInt(state.outString(i.i));} catch(...) {return Integer::NAelement;}} }

template<>
static Double::Element Cast<Character, Double>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Double::NAelement; else {try{return strToDouble(state.outString(i.i));} catch(...) {return Double::NAelement;}} }

template<>
static Complex::Element Cast<Character, Complex>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Complex::NAelement; else {try{return strToComplex(state.outString(i.i));} catch(...) {return Complex::NAelement;}} }

template<>
static List::Element Cast<Character, List>(State& state, Character::Element const& i) { return Character::c(i); }


template<>
static Logical::Element Cast<List, Logical>(State& state, List::Element const& i) { _error("NYI list to vector cast"); }

template<>
static Integer::Element Cast<List, Integer>(State& state, List::Element const& i) { _error("NYI list to vector cast"); }

template<>
static Double::Element Cast<List, Double>(State& state, List::Element const& i) { _error("NYI list to vector cast"); }

template<>
static Complex::Element Cast<List, Complex>(State& state, List::Element const& i) { _error("NYI list to vector cast"); }

template<>
static Character::Element Cast<List, Character>(State& state, List::Element const& i) { _error("NYI list to vector cast"); }



template<class I, class O> 
struct CastOp : public UnaryOp<I, O> {
	static typename CastOp::R eval(State& state, typename CastOp::A const& i) { return Cast<typename CastOp::AV, typename CastOp::RV>(state, i); }
};

template<class O>
O As(State& state, Value const& src) {
	if(src.type == O::type)
		return src;
	switch(src.type.Enum()) {
		case Type::E_R_double: return Zip1< CastOp<Double, O> >::eval(state, src); break;
		case Type::E_R_integer: return Zip1< CastOp<Integer, O> >::eval(state, src); break;
		case Type::E_R_logical: return Zip1< CastOp<Logical, O> >::eval(state, src); break;
		case Type::E_R_complex: return Zip1< CastOp<Complex, O> >::eval(state, src); break;
		case Type::E_R_character: return Zip1< CastOp<Character, O> >::eval(state, src); break;
		case Type::E_R_list: return Zip1< CastOp<List, O> >::eval(state, src); break;
		default: _error("Invalid cast"); break;
	};
}

inline Value As(State& state, Type type, Value const& src) {
	if(src.type == type)
		return src;
	switch(type.Enum()) {
		case Type::E_R_double: return As<Double>(state, src); break;
		case Type::E_R_integer: return As<Integer>(state, src); break;
		case Type::E_R_logical: return As<Logical>(state, src); break;
		case Type::E_R_complex: return As<Complex>(state, src); break;
		case Type::E_R_character: return As<Character>(state, src); break;
		case Type::E_R_list: return As<List>(state, src); break;
		default: _error("Invalid cast"); break;
	};
}


void importCoerceFunctions(State& state);

#endif
