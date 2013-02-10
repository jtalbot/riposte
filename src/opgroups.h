
#ifndef _RIPOSTE_OPGROUPS_H
#define _RIPOSTE_OPGROUPS_H

#include "vector.h"

template<class X> struct ArithUnary1   { typedef X A; typedef Double MA; typedef Double R; };
template<> struct ArithUnary1<Logical> { typedef Logical A; typedef Integer MA; typedef Integer R; };
template<> struct ArithUnary1<Integer> { typedef Integer A; typedef Integer MA; typedef Integer R; };

template<class X> struct ArithUnary2   { typedef X A; typedef Double MA; typedef Double R; };

template<class X> struct LogicalUnary  { typedef X A; typedef Logical MA; typedef Logical R; };

template<class X> struct OrdinalUnary  { typedef X A; typedef X MA; typedef Logical R; };



template<class X, class Y> struct ArithBinary1 
    { typedef X A; typedef Y B; typedef Double MA; typedef Double MB; typedef Double R; };
template<> struct ArithBinary1<Logical,Logical> 
    { typedef Logical A; typedef Logical B; typedef Integer MA; typedef Integer MB; typedef Integer R; };
template<> struct ArithBinary1<Logical,Integer> 
    { typedef Logical A; typedef Integer B; typedef Integer MA; typedef Integer MB; typedef Integer R; };
template<> struct ArithBinary1<Integer,Logical> 
    { typedef Integer A; typedef Logical B; typedef Integer MA; typedef Integer MB; typedef Integer R; };
template<> struct ArithBinary1<Integer,Integer> 
    { typedef Integer A; typedef Integer B; typedef Integer MA; typedef Integer MB; typedef Integer R; };

template<class X, class Y> struct ArithBinary2 
    { typedef X A; typedef Y B; typedef Double MA; typedef Double MB; typedef Double R; };

template<class X, class Y> struct LogicalBinary
    { typedef X A; typedef Y B; typedef Logical MA; typedef Logical MB; typedef Logical R; };

template<class X, class Y> struct RoundBinary
    { typedef X A; typedef Y B; typedef Double MA; typedef Integer MB; typedef Double R; };


template<class X, class Y> struct OrdinalBinary {};
#define ORDINAL_BINARY(X, Y, Z) \
    template<> struct OrdinalBinary<X, Y> \
    { typedef X A; typedef Y B; typedef Z MA; typedef Z MB; typedef Logical R; };
DEFAULT_TYPE_MEET(ORDINAL_BINARY)
#undef ORDINAL_BINARY

template<class X, class Y> struct UnifyBinary {};
#define UNIFY_BINARY(X, Y, Z) \
    template<> struct UnifyBinary<X, Y> \
    { typedef X A; typedef Y B; typedef Z MA; typedef Z MB; typedef Z R; };
DEFAULT_TYPE_MEET(UNIFY_BINARY)
#undef UNIFY_BINARY


template<class X> struct ArithFold1   { typedef X A; typedef Double MA; typedef Double R; };
template<> struct ArithFold1<Logical> { typedef Logical A; typedef Integer MA; typedef Integer R; };
template<> struct ArithFold1<Integer> { typedef Integer A; typedef Integer MA; typedef Integer R; };

template<class X> struct ArithFold2   { typedef X A; typedef Double MA; typedef Double R; };

template<class X> struct UnifyFold { typedef X A; typedef X MA; typedef X R; };

template<class X> struct LogicalFold { typedef X A; typedef Logical MA; typedef Logical R; };

// Some special folds that we need to formalize
template<class X> struct CountFold   { typedef X A; typedef X MA; typedef Integer R; };
template<class X, class Y> struct Moment2Fold { 
    typedef X A; typedef Y B; typedef Double MA; typedef Double MB; typedef Double R; };


template<class X> struct ArithScan   { typedef X A; typedef Double MA; typedef Double R; };
template<> struct ArithScan<Logical> { typedef Logical A; typedef Integer MA; typedef Integer R; };
template<> struct ArithScan<Integer> { typedef Integer A; typedef Integer MA; typedef Integer R; };

template<class X> struct UnifyScan { typedef X A; typedef X MA; typedef X R; };




// More complicated ops
template<class X, class Y, class Z> struct IfElse  { 
    typedef X A; typedef Y B; typedef Z C; 
    typedef Logical MA; typedef typename UnifyBinary<Y, Z>::MA MB; typedef typename UnifyBinary<Y, Z>::MB MC;
    typedef typename UnifyBinary<Y, Z>::R R; 
};

template<class X, class Y> struct Split
    { typedef X A; typedef Y B; typedef Integer MA; typedef Y MB; typedef Y R; };

#endif

