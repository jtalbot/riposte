
#ifndef _RIPOSTE_COERCE_H
#define _RIPOSTE_COERCE_H

#include "value.h"
#include "vector.h"
#include "interpreter.h"

#include <limits>


// Casting between scalar types
template<typename I, typename O>
static void Cast1(I const& i, O& o) { o = (O const&)i; }

// Casting functions between vector types
template<typename I, typename O>
static typename O::Element Cast(typename I::Element const& i) { return (typename O::Element)i; }

template<class I, class O> 
struct CastOp {
	typedef I A;
	typedef O R;
	static typename O::Element eval(void* args, typename I::Element i) { return Cast<I, O>(i); }
	static void Scalar(void* args, typename I::Element i, Value& c) { R::InitScalar(c, eval(args, i)); }
};

template<class O>
void As(Value src, O& out) {
	if(src.type() == O::ValueType) {
		out = (O const&)src;
		return;
	}
	switch(src.type()) {
		case Type::Null: O(0); return; break;
		#define CASE(Name,...) case Type::Name: Zip1< CastOp<Name, O> >::eval(nullptr, (Name const&)src, out); return; break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		case Type::Closure:
			Cast1<Closure, O>((Closure const&)src, out); return; break;
		default: break;
	};
	out.attributes(((Object const&)src).attributes());
	_error(std::string("Invalid cast from ") + Type::toString(src.type()) + " to " + Type::toString(O::ValueType));
}

template<>
inline void As<Null>(Value src, Null& out) {
       out = Null::Singleton();
}

template<class O>
O As(Value const& src) {
	O out;
	As<O>(src, out);
	return out;
}

inline Value As(Type::Enum type, Value const& src) {
	if(src.type() == type)
		return src;
	switch(type) {
		case Type::Null: return Null::Singleton(); break;
		#define CASE(Name,...) case Type::Name: return As<Name>(src); break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		default: _error(std::string("Invalid cast from ") + Type::toString(src.type()) + " to " + Type::toString(type)); break;
	};
}


template<>
SPECIALIZED_STATIC Raw::Element Cast<Logical, Raw>(Logical::Element const& i) { return Logical::isTrue(i) ? 1 : 0; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Logical, Integer>(Logical::Element const& i) { return Logical::isNA(i) ? Integer::NAelement : i ? 1 : 0; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Logical, Double>(Logical::Element const& i) { return Logical::isNA(i) ? Double::NAelement : i ? 1.0 : 0.0; }

template<>
SPECIALIZED_STATIC Character::Element Cast<Logical, Character>(Logical::Element const& i) { return Logical::isNA(i) ? Character::NAelement : i ? Strings::True : Strings::False; }

template<>
SPECIALIZED_STATIC List::Element Cast<Logical, List>(Logical::Element const& i) { return Logical::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Integer, Raw>(Integer::Element const& i) { return (Raw::Element)i; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Integer, Logical>(Integer::Element const& i) { return Integer::isNA(i) ? Logical::NAelement : i != 0 ? Logical::TrueElement : Logical::FalseElement; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Integer, Double>(Integer::Element const& i) { return Integer::isNA(i) ? Double::NAelement : (Double::Element)i; }

template<>
SPECIALIZED_STATIC Character::Element Cast<Integer, Character>(Integer::Element const& i) { return Integer::isNA(i) ? Character::NAelement : MakeString(intToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Integer, List>(Integer::Element const& i) { return Integer::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Double, Raw>(Double::Element const& i) { return (Raw::Element)i; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Double, Logical>(Double::Element const& i) { return Double::isNA(i) ? Logical::NAelement : i != 0.0 ? Logical::TrueElement : Logical::FalseElement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Double, Integer>(Double::Element const& i) { return Double::isNA(i) || i > std::numeric_limits<Integer::Element>::max() || i < std::numeric_limits<Integer::Element>::min() ? Integer::NAelement : i;  }

template<>
SPECIALIZED_STATIC Character::Element Cast<Double, Character>(Double::Element const& i) { return Integer::isNA(i) ? Character::NAelement : MakeString(doubleToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Double, List>(Double::Element const& i) { return Double::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Character, Raw>(Character::Element const& i) { return 0; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Character, Logical>(Character::Element const& i) { if(Eq(i, Strings::True)) return Logical::TrueElement; else if(Eq(i, Strings::False)) return Logical::FalseElement; else return Logical::NAelement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Character, Integer>(Character::Element const& i) { if(Character::isNA(i)) return Integer::NAelement; else {try{return strToInt(i->s);} catch(...) {return Integer::NAelement;}} }

template<>
SPECIALIZED_STATIC Double::Element Cast<Character, Double>(Character::Element const& i) { if(Character::isNA(i)) return Double::NAelement; else {try{return strToDouble(i->s);} catch(...) {return Double::NAelement;}} }

template<>
SPECIALIZED_STATIC List::Element Cast<Character, List>(Character::Element const& i) { return Character::c(i); }


template<>
SPECIALIZED_STATIC Character::Element Cast<Raw, Character>(Raw::Element const& i) { return MakeString(rawToStr(i)); }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Raw, Logical>(Raw::Element const& i) { return i ? Logical::TrueElement : Logical::FalseElement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Raw, Integer>(Raw::Element const& i) { return i; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Raw, Double>(Raw::Element const& i) { return i; }

template<>
SPECIALIZED_STATIC List::Element Cast<Raw, List>(Raw::Element const& i) { return Raw::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<List, Raw>(List::Element const& i) { Raw a = As<Raw>(i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to raw"); }

template<>
SPECIALIZED_STATIC Logical::Element Cast<List, Logical>(List::Element const& i) { Logical a = As<Logical>(i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to logical"); }

template<>
SPECIALIZED_STATIC Integer::Element Cast<List, Integer>(List::Element const& i) { Integer a = As<Integer>(i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to integer"); }

template<>
SPECIALIZED_STATIC Double::Element Cast<List, Double>(List::Element const& i) { Double a = As<Double>(i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to double"); }

template<>
SPECIALIZED_STATIC Character::Element Cast<List, Character>(List::Element const& i) { Character a = As<Character>(i); if(a.length()==1) return a[0]; else _error("Invalid cast from list to character"); }



template<>
SPECIALIZED_STATIC void Cast1<Closure, List>(Closure const& i, List& o) { List::InitScalar(o, (Value const&)i); }


#endif

