
#ifndef _RIPOSTE_COERCE_H
#define _RIPOSTE_COERCE_H

#include "value.h"
#include "vector.h"
#include "interpreter.h"

#include <limits>


// Casting between scalar types
template<typename I, typename O>
static void Cast1(Thread& thread, I const& i, O& o) { o = (O const&)i; }

// Casting functions between vector types
template<typename I, typename O>
static typename O::Element Cast(Thread& thread, typename I::Element const& i) { return (typename O::Element)i; }

template<class I, class O> 
struct CastOp {
	typedef I A;
	typedef O R;
	static typename O::Element eval(Thread& thread, typename I::Element i) { return Cast<I, O>(thread, i); }
	static void Scalar(Thread& thread, typename I::Element i, Value& c) { R::InitScalar(c, eval(thread, i)); }
};

template<class O>
void As(Thread& thread, Value src, O& out) {
	if(src.isObject())
		src = ((Object const&)src).base();

	if(src.type == O::VectorType) {
		out = (O const&)src;
		return;
	}
	switch(src.type) {
		case Type::Null: O(0); return; break;
		#define CASE(Name,...) case Type::Name: Zip1< CastOp<Name, O> >::eval(thread, (Name const&)src, out); return; break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		case Type::Function:
			Cast1<Function, O>(thread, (Function const&)src, out); return; break;
		default: break;
	};
	_error(std::string("Invalid cast from ") + Type::toString(src.type) + " to " + Type::toString(O::VectorType));
}

template<>
inline void As<Null>(Thread& thread, Value src, Null& out) {
       out = Null::Singleton();
}

template<class O>
O As(Thread& thread, Value const& src) {
	O out;
	As<O>(thread, src, out);
	return out;
}

inline Value As(Thread& thread, Type::Enum type, Value const& src) {
	if(src.type == type)
		return src;
	switch(type) {
		case Type::Null: return Null::Singleton(); break;
		#define CASE(Name,...) case Type::Name: return As<Name>(thread, src); break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		default: _error(std::string("Invalid cast from ") + Type::toString(src.type) + " to " + Type::toString(type)); break;
	};
}



//gcc >= 4.3 requires template specialization to have the same storage class (e.g. 'static') as the orignal template
//specifying static is an error
//gcc < 4.3 treats the declarations as distinct. Not specifying a storage class makes the specialization external >_<
//#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
#define SPECIALIZED_STATIC
//#else
//#define SPECIALIZED_STATIC static
//#endif



template<>
SPECIALIZED_STATIC Raw::Element Cast<Logical, Raw>(Thread& thread, Logical::Element const& i) { return Logical::isTrue(i) ? 1 : 0; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Logical, Integer>(Thread& thread, Logical::Element const& i) { return Logical::isNA(i) ? Integer::NAelement : i ? 1 : 0; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Logical, Double>(Thread& thread, Logical::Element const& i) { return Logical::isNA(i) ? Double::NAelement : i ? 1.0 : 0.0; }

template<>
SPECIALIZED_STATIC Character::Element Cast<Logical, Character>(Thread& thread, Logical::Element const& i) { return Logical::isNA(i) ? Character::NAelement : i ? Strings::True : Strings::False; }

template<>
SPECIALIZED_STATIC List::Element Cast<Logical, List>(Thread& thread, Logical::Element const& i) { return Logical::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Integer, Raw>(Thread& thread, Integer::Element const& i) { return (Raw::Element)i; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Integer, Logical>(Thread& thread, Integer::Element const& i) { return Integer::isNA(i) ? Logical::NAelement : i != 0 ? Logical::TrueElement : Logical::FalseElement; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Integer, Double>(Thread& thread, Integer::Element const& i) { return Integer::isNA(i) ? Double::NAelement : (Double::Element)i; }

template<>
SPECIALIZED_STATIC Character::Element Cast<Integer, Character>(Thread& thread, Integer::Element const& i) { return Integer::isNA(i) ? Character::NAelement : thread.internStr(intToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Integer, List>(Thread& thread, Integer::Element const& i) { return Integer::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Double, Raw>(Thread& thread, Double::Element const& i) { return (Raw::Element)i; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Double, Logical>(Thread& thread, Double::Element const& i) { return Double::isNA(i) ? Logical::NAelement : i != 0.0 ? Logical::TrueElement : Logical::FalseElement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Double, Integer>(Thread& thread, Double::Element const& i) { return Double::isNA(i) || i > std::numeric_limits<Integer::Element>::max() || i < std::numeric_limits<Integer::Element>::min() ? Integer::NAelement : i;  }

template<>
SPECIALIZED_STATIC Character::Element Cast<Double, Character>(Thread& thread, Double::Element const& i) { return Integer::isNA(i) ? Character::NAelement : thread.internStr(doubleToStr(i)); }

template<>
SPECIALIZED_STATIC List::Element Cast<Double, List>(Thread& thread, Double::Element const& i) { return Double::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<Character, Raw>(Thread& thread, Character::Element const& i) { return 0; }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Character, Logical>(Thread& thread, Character::Element const& i) { if(i == Strings::True) return Logical::TrueElement; else if(i == Strings::False) return Logical::FalseElement; else return Logical::NAelement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Character, Integer>(Thread& thread, Character::Element const& i) { if(Character::isNA(i)) return Integer::NAelement; else {try{return strToInt(thread.externStr(i));} catch(...) {return Integer::NAelement;}} }

template<>
SPECIALIZED_STATIC Double::Element Cast<Character, Double>(Thread& thread, Character::Element const& i) { if(Character::isNA(i)) return Double::NAelement; else {try{return strToDouble(thread.externStr(i));} catch(...) {return Double::NAelement;}} }

template<>
SPECIALIZED_STATIC List::Element Cast<Character, List>(Thread& thread, Character::Element const& i) { return Character::c(i); }


template<>
SPECIALIZED_STATIC Character::Element Cast<Raw, Character>(Thread& thread, Raw::Element const& i) { return thread.internStr(rawToStr(i)); }

template<>
SPECIALIZED_STATIC Logical::Element Cast<Raw, Logical>(Thread& thread, Raw::Element const& i) { return i ? Logical::TrueElement : Logical::FalseElement; }

template<>
SPECIALIZED_STATIC Integer::Element Cast<Raw, Integer>(Thread& thread, Raw::Element const& i) { return i; }

template<>
SPECIALIZED_STATIC Double::Element Cast<Raw, Double>(Thread& thread, Raw::Element const& i) { return i; }

template<>
SPECIALIZED_STATIC List::Element Cast<Raw, List>(Thread& thread, Raw::Element const& i) { return Raw::c(i); }


template<>
SPECIALIZED_STATIC Raw::Element Cast<List, Raw>(Thread& thread, List::Element const& i) { Raw a = As<Raw>(thread, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to raw"); }

template<>
SPECIALIZED_STATIC Logical::Element Cast<List, Logical>(Thread& thread, List::Element const& i) { Logical a = As<Logical>(thread, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to logical"); }

template<>
SPECIALIZED_STATIC Integer::Element Cast<List, Integer>(Thread& thread, List::Element const& i) { Integer a = As<Integer>(thread, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to integer"); }

template<>
SPECIALIZED_STATIC Double::Element Cast<List, Double>(Thread& thread, List::Element const& i) { Double a = As<Double>(thread, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to double"); }

template<>
SPECIALIZED_STATIC Character::Element Cast<List, Character>(Thread& thread, List::Element const& i) { Character a = As<Character>(thread, i); if(a.length==1) return a[0]; else _error("Invalid cast from list to character"); }



template<>
SPECIALIZED_STATIC void Cast1<Function, List>(Thread& thread, Function const& i, List& o) { List::InitScalar(o, (Value const&)i); }


#endif

