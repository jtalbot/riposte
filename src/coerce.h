
#ifndef _RIPOSTE_COERCE_H
#define _RIPOSTE_COERCE_H

#include "value.h"
#include "vector.h"

#include <limits>


// Casting between scalar types
template<typename I, typename O>
static void Cast1(State& state, I const& i, O& o) { o = (O)i; }

// Casting functions between vector types
template<typename I, typename O>
static typename O::Element Cast(State& state, typename I::Element const& i) { return (typename O::Element)i; }

template<class I, class O> 
struct CastOp : public UnaryOp<I, O> {
	static typename CastOp::R eval(State& state, typename CastOp::A const& i) { return Cast<typename CastOp::AV, typename CastOp::RV>(state, i); }
};

template<class O>
void As(State& state, Value const& src, O& out) {
	if(src.type == O::VectorType)
		out = (O const&)src;
	switch(src.type) {
		case Type::Null: O(0); return; break;
		#define CASE(Name,...) case Type::Name: Zip1< CastOp<Name, O> >::eval(state, (Name const&)src, out); return; break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		case Type::Symbol: 
			Cast1<Symbol, O>(state, (Symbol const&)src, out); return; break;
		case Type::Function:
			Cast1<Function, O>(state, (Function const&)src, out); return; break;
		default: break;
	};
	_error(std::string("Invalid cast from ") + Type::toString(src.type) + " to " + Type::toString(O::VectorType));
}

template<>
inline void As<Null>(State& state, Value const& src, Null& out) {
       out = Null::Singleton();
}

template<class O>
O As(State& state, Value const& src) {
	O out;
	As<O>(state, src, out);
	return O(out);
}

inline Value As(State& state, Type::Enum type, Value const& src) {
	if(src.type == type)
		return src;
	switch(type) {
		case Type::Null: return Null::Singleton(); break;
		#define CASE(Name,...) case Type::Name: return As<Name>(state, src); break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		default: _error(std::string("Invalid cast from ") + Type::toString(src.type) + " to " + Type::toString(type)); break;
	};
}



//gcc >= 4.3 requires template specialization to have the same storage class (e.g. 'static') as the orignal template
//specifying static is an error
//gcc < 4.3 treats the declarations as distinct. Not specifying a storage class makes the specialization external >_<
#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
#define SPECIALIZED_STATIC
#else
#define SPECIALIZED_STATIC static
#endif



template<>
SPECIALIZED_STATIC Raw::Element Cast<Logical, Raw>(State& state, Logical::Element const& i) { return i; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Logical, Integer>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Integer::NAelement : i ? 1 : 0; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Logical, Double>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Double::NAelement : i ? 1.0 : 0.0; }

template<>
SPECIALIZED_STATIC Character::Element Cast<Logical, Character>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Character::NAelement : i ? Strings::TRUE : Strings::FALSE; }

template<>
SPECIALIZED_STATIC List::Element Cast<Logical, List>(State& state, Logical::Element const& i) { return Logical::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Integer, Raw>(State& state, Integer::Element const& i) { return (Raw::Element)i; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Integer, Logical>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Logical::NAelement : i != 0 ? 1 : 0; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Integer, Double>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Double::NAelement : (Double::Element)i; }

template<>
SPECIALIZED_STATIC Character::Element Cast<Integer, Character>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Character::NAelement : state.internStr(intToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Integer, List>(State& state, Integer::Element const& i) { return Integer::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Double, Raw>(State& state, Double::Element const& i) { return (Raw::Element)i; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Double, Logical>(State& state, Double::Element const& i) { return Double::isNA(i) ? Logical::NAelement : i != 0.0 ? 1 : 0; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Double, Integer>(State& state, Double::Element const& i) { return Double::isNA(i) || i > std::numeric_limits<Integer::Element>::max() || i < std::numeric_limits<Integer::Element>::min() ? Integer::NAelement : i;  }

template<>
SPECIALIZED_STATIC Character::Element Cast<Double, Character>(State& state, Double::Element const& i) { return Integer::isNA(i) ? Character::NAelement : state.internStr(doubleToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Double, List>(State& state, Double::Element const& i) { return Double::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Character, Raw>(State& state, Character::Element const& i) { return 0; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Character, Logical>(State& state, Character::Element const& i) { if(i == Strings::TRUE) return 1; else if(i == Strings::FALSE) return 0; else return Logical::NAelement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Character, Integer>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Integer::NAelement; else {try{return strToInt(state.externStr(i));} catch(...) {return Integer::NAelement;}} }

template<>
SPECIALIZED_STATIC Double::Element Cast<Character, Double>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Double::NAelement; else {try{return strToDouble(state.externStr(i));} catch(...) {return Double::NAelement;}} }

template<>
SPECIALIZED_STATIC List::Element Cast<Character, List>(State& state, Character::Element const& i) { return Character::c(i); }


template<>
SPECIALIZED_STATIC Character::Element Cast<Raw, Character>(State& state, Raw::Element const& i) { return state.internStr(rawToStr(i)); }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Raw, Logical>(State& state, Raw::Element const& i) { return (Logical::Element)i; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Raw, Integer>(State& state, Raw::Element const& i) { return i; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Raw, Double>(State& state, Raw::Element const& i) { return i; }

template<>
SPECIALIZED_STATIC List::Element Cast<Raw, List>(State& state, Raw::Element const& i) { return Raw::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<List, Raw>(State& state, List::Element const& i) { Raw a = As<Raw>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to raw"); }

template<>
SPECIALIZED_STATIC Logical::Element Cast<List, Logical>(State& state, List::Element const& i) { Logical a = As<Logical>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to logical"); }

template<>
SPECIALIZED_STATIC Integer::Element Cast<List, Integer>(State& state, List::Element const& i) { Integer a = As<Integer>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to integer"); }

template<>
SPECIALIZED_STATIC Double::Element Cast<List, Double>(State& state, List::Element const& i) { Double a = As<Double>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to double"); }

template<>
SPECIALIZED_STATIC Character::Element Cast<List, Character>(State& state, List::Element const& i) { Character a = As<Character>(state, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to character"); }



// Symbol casting...move somewhere else?
template<>
SPECIALIZED_STATIC void Cast1<Symbol, Raw>(State& state, Symbol const& i, Raw& o) { _error("Invalid cast from symbol to raw"); }

template<>
SPECIALIZED_STATIC void Cast1<Symbol, Logical>(State& state, Symbol const& i, Logical& o) { _error("Invalid cast from symbol to logical"); }

template<>
SPECIALIZED_STATIC void Cast1<Symbol, Integer>(State& state, Symbol const& i, Integer& o) { _error("Invalid cast from symbol to integer"); }

template<>
SPECIALIZED_STATIC void Cast1<Symbol, Double>(State& state, Symbol const& i, Double& o) { _error("Invalid cast from symbol to double"); }

template<>
SPECIALIZED_STATIC void Cast1<Symbol, Character>(State& state, Symbol const& i, Character& o) { Character::InitScalar(o, i); }

template<>
SPECIALIZED_STATIC void Cast1<Symbol, List>(State& state, Symbol const& i, List& o) { List::InitScalar(o, i); }


template<>
SPECIALIZED_STATIC void Cast1<Function, List>(State& state, Function const& i, List& o) { List::InitScalar(o, (Value const&)i); }


#endif

