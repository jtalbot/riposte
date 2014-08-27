
#ifndef _RIPOSTE_COERCE_H
#define _RIPOSTE_COERCE_H

#include "value.h"
#include "vector.h"
#include "interpreter.h"

#include <limits>


// Casting between scalar types
template<typename I, typename O>
static void Cast1(State& state, I const& i, O& o) { o = (O const&)i; }

// Casting functions between vector types
template<typename I, typename O>
static typename O::Element Cast(State& state, typename I::Element const& i) { return (typename O::Element)i; }

template<class I, class O> 
struct CastOp {
	typedef I A;
	typedef O R;
	static typename O::Element eval(State& state, void* args, typename I::Element i) { return Cast<I, O>(state, i); }
	static void Scalar(State& state, void* args, typename I::Element i, Value& c) { R::InitScalar(c, eval(state, args, i)); }
};

template<class O>
void As(State& state, Value src, O& out) {
	if(src.type() == O::ValueType) {
		out = (O const&)src;
		return;
	}
	switch(src.type()) {
		case Type::Null: O(0); return; break;
		#define CASE(Name,...) case Type::Name: Zip1< CastOp<Name, O> >::eval(state, NULL, (Name const&)src, out); return; break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		case Type::Closure:
			Cast1<Closure, O>(state, (Closure const&)src, out); return; break;
		default: break;
	};
	out.attributes(((Object const&)src).attributes());
	_error(std::string("Invalid cast from ") + Type::toString(src.type()) + " to " + Type::toString(O::ValueType));
}

template<>
inline void As<Null>(State& state, Value src, Null& out) {
       out = Null::Singleton();
}

template<class O>
O As(State& state, Value const& src) {
	O out;
	As<O>(state, src, out);
	return out;
}

inline Value As(State& state, Type::Enum type, Value const& src) {
	if(src.type() == type)
		return src;
	switch(type) {
		case Type::Null: return Null::Singleton(); break;
		#define CASE(Name,...) case Type::Name: return As<Name>(state, src); break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		default: _error(std::string("Invalid cast from ") + Type::toString(src.type()) + " to " + Type::toString(type)); break;
	};
}


template<>
SPECIALIZED_STATIC Raw::Element Cast<Logical, Raw>(State& state, Logical::Element const& i) { return Logical::isTrue(i) ? 1 : 0; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Logical, Integer>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Integer::NAelement : i ? 1 : 0; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Logical, Double>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Double::NAelement : i ? 1.0 : 0.0; }

template<>
SPECIALIZED_STATIC Character::Element Cast<Logical, Character>(State& state, Logical::Element const& i) { return Logical::isNA(i) ? Character::NAelement : i ? Strings::True : Strings::False; }

template<>
SPECIALIZED_STATIC List::Element Cast<Logical, List>(State& state, Logical::Element const& i) { return Logical::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Integer, Raw>(State& state, Integer::Element const& i) { return (Raw::Element)i; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Integer, Logical>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Logical::NAelement : i != 0 ? Logical::TrueElement : Logical::FalseElement; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Integer, Double>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Double::NAelement : (Double::Element)i; }

template<>
SPECIALIZED_STATIC Character::Element Cast<Integer, Character>(State& state, Integer::Element const& i) { return Integer::isNA(i) ? Character::NAelement : state.internStr(intToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Integer, List>(State& state, Integer::Element const& i) { return Integer::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Double, Raw>(State& state, Double::Element const& i) { return (Raw::Element)i; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Double, Logical>(State& state, Double::Element const& i) { return Double::isNA(i) ? Logical::NAelement : i != 0.0 ? Logical::TrueElement : Logical::FalseElement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Double, Integer>(State& state, Double::Element const& i) { return Double::isNA(i) || i > std::numeric_limits<Integer::Element>::max() || i < std::numeric_limits<Integer::Element>::min() ? Integer::NAelement : i;  }

template<>
SPECIALIZED_STATIC Character::Element Cast<Double, Character>(State& state, Double::Element const& i) { return Integer::isNA(i) ? Character::NAelement : state.internStr(doubleToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Double, List>(State& state, Double::Element const& i) { return Double::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Character, Raw>(State& state, Character::Element const& i) { return 0; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Character, Logical>(State& state, Character::Element const& i) { if(i == Strings::True) return Logical::TrueElement; else if(i == Strings::False) return Logical::FalseElement; else return Logical::NAelement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Character, Integer>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Integer::NAelement; else {try{return strToInt(state.externStr(i));} catch(...) {return Integer::NAelement;}} }

template<>
SPECIALIZED_STATIC Double::Element Cast<Character, Double>(State& state, Character::Element const& i) { if(Character::isNA(i)) return Double::NAelement; else {try{return strToDouble(state.externStr(i));} catch(...) {return Double::NAelement;}} }

template<>
SPECIALIZED_STATIC List::Element Cast<Character, List>(State& state, Character::Element const& i) { return Character::c(i); }


template<>
SPECIALIZED_STATIC Character::Element Cast<Raw, Character>(State& state, Raw::Element const& i) { return state.internStr(rawToStr(i)); }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Raw, Logical>(State& state, Raw::Element const& i) { return i ? Logical::TrueElement : Logical::FalseElement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Raw, Integer>(State& state, Raw::Element const& i) { return i; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Raw, Double>(State& state, Raw::Element const& i) { return i; }

template<>
SPECIALIZED_STATIC List::Element Cast<Raw, List>(State& state, Raw::Element const& i) { return Raw::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<List, Raw>(State& state, List::Element const& i) { Raw a = As<Raw>(state, i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to raw"); }

template<>
SPECIALIZED_STATIC Logical::Element Cast<List, Logical>(State& state, List::Element const& i) { Logical a = As<Logical>(state, i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to logical"); }

template<>
SPECIALIZED_STATIC Integer::Element Cast<List, Integer>(State& state, List::Element const& i) { Integer a = As<Integer>(state, i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to integer"); }

template<>
SPECIALIZED_STATIC Double::Element Cast<List, Double>(State& state, List::Element const& i) { Double a = As<Double>(state, i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to double"); }

template<>
SPECIALIZED_STATIC Character::Element Cast<List, Character>(State& state, List::Element const& i) { Character a = As<Character>(state, i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to character"); }



template<>
SPECIALIZED_STATIC void Cast1<Closure, List>(State& state, Closure const& i, List& o) { List::InitScalar(o, (Value const&)i); }


#endif

