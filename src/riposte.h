
#ifndef RIPOSTE_H
#define RIPOSTE_H

#include <string>
#include <stdint.h>

/* The Riposte C++ API */

namespace Riposte {

// Riposte initialization and finalization
enum Format {
    RiposteFormat,
    RFormat
};

//      Initialize Riposte global state. Must be called first.
void        initialize(int argc, char** argv,
                int threads, bool verbose, Format format, bool profile);

//      Clean up the Riposte global state. Must be called last.
void        finalize();


// Riposte execution state
class       State;

//      Generate a new execution state which holds the internal
//      Riposte VM data structures. Each execution thread should
//      use its own state. State is a relatively heavy-weight 
//      object, so try to reuse when possible.
State&      newState        ();

//      Delete an execution state, freeing the memory used
//      by the VM data structures.
void        deleteState     (State&);


// String handling
struct StringImpl;
typedef const StringImpl* String;

//      Intern an stl string.
String		newString    (State const&, std::string const& str);
//      Convert back to an stl string.
std::string getString    (State const&, String str);

/*
// An opaque fat pointer to a Riposte value. Users can assume 
// that a Value is 128 bits, but shouldn't make any assumptions
// about its contents.
class Value {
private:
    uint64_t dontUseMe[2];
public:

    Value       get         (Value const& index) const;
    Value       set         (Value const& index, Value const& v) const;
    Value       getSlice    (Value const& index) const;
    Value       setSlice    (Value const& index, Value const& v) const;
};


// Represents a reference type.
// All references to the same value see the same updates.

// a <- ref(c(1,2,3))
// b <- a  (b holds the reference)
// d <- deref(a)   (d holds a copy of c(1,2,3))

// deref(b) <- 10
// a and b both now see c(10)
// d still sees c(1,2,3)

// `deref<-` (b, 10)

// x <- deref(b) + 1

class Reference : public Value {
public:
    Reference               (Value target);
    Value      deref        () const;
};


class Vector : public Value {
public:
    int64_t     length      () const;
};


class Unit : public Vector {
public:
    Unit                (int64_t length);
   
    void     get        (int64_t index) const;
    void     set        (int64_t index);
    
    Unit     getSlice   (Integer index) const;
    Unit     setSlice   (Integer index) const;
};


class Raw : public Vector {
public:
    Raw (int64_t length);
    
    Just<uint8_t>   get (int64_t index) const;
    void            set (int64_t index, Just<uint8_t> v);
};


class Logical : public Vector {
public:
    Logical (int64_t length);

    Maybe<bool>     get (int64_t index) const;
    void            set (int64_t index, Maybe<bool> v);
    
    Logical     getSlice    (Integer const& index) const;
    Logical     setSlice    (Integer const& index, Logical const& v) const;
};


class Integer : public Vector {
public:
    Integer (int64_t length);

    Maybe<int64_t>  get (int64_t index) const;
    void            set (int64_t index, Maybe<int64_t> v);

    Integer     getSlice    (Integer index) const;
    Integer     setSlice    (Integer index, Integer v) const;
};


class Double : public Vector {
public:
    Double (int64_t length);
    
    Maybe<double>   get (int64_t index) const;
    void            set (int64_t index, Maybe<double> v);
};


class Complex : public Vector {
public:
    Complex (int64_t length);
    
    Maybe<complex double> get(int64_t index) const;
    void                  set (int64_t index, Maybe<complex double> v);
};


class Character : public Vector {
public:
    String          get (int64_t index) const;
    void            set (int64_t index, String v);
};


class Pointer : public Vector {
public:
    void*           get (int64_t index) const;
    void            set (int64_t index, void* v);
};


class List : public Vector {
public:
    Value       get         (int64_t index) const;
    void        set         (int64_t index, Value v);
    
    List        getSlice    (Integer index) const;
    List        setSlice    (Integer index, List v) const;
};

// Environment
class Environment : public Value {
public:
    Value       get         (String index) const;
    void        set         (String index, Value v);
    
    Environment getSlice    (Character index) const;
    Environment setSlice    (Character index, List v) const;
};

// Closure
class Closure : public Value {
public:
    Value       get         (List arguments) const;
};


// Represents any value in R that has attributes.
// Strip produces the base object without any attributes.
class Object : public Value {
public:
    Object                  (Value base);
    Value       getSlot     (String slot) const;
    Object      setSlot     (String slot, Value v) const;
    Value       slots       () const;
    Value       strip       () const;
};
*/


// C-style API

/*
// Convert Riposte values to C values
// Index must be in bounds already (does not automatically grow)
void        getNull         (State const&, Value const&, int64_t index);
char        getRaw          (State const&, Value const&, int64_t index);
bool        getLogical      (State const&, Value const&, int64_t index);
int64_t     getInteger      (State const&, Value const&, int64_t index);
double      getDouble       (State const&, Value const&, int64_t index);
String      getCharacter    (State const&, Value const&, int64_t index);
void*       getPointer      (State const&, Value const&, int64_t index);

// Set Riposte values from C values
// Index must be in bounds already (does not automatically grow)
Value       setNull         (State const&, Value const&, int64_t index);
Value       setRaw          (State const&, Value const&, int64_t index, char);
Value       setLogical      (State const&, Value const&, int64_t index, bool);
Value       setInteger      (State const&, Value const&, int64_t index, int64_t);
Value       setDouble       (State const&, Value const&, int64_t index, double);
Value       setCharacter    (State const&, Value const&, int64_t index, String);
Value       setPointer      (State const&, Value const&, int64_t index, void*);

// Check for NAs
bool        isnaNull        (State const&, Value const&, int64_t index);
bool        isnaRaw         (State const&, Value const&, int64_t index);
bool        isnaLogical     (State const&, Value const&, int64_t index);
bool        isnaInteger     (State const&, Value const&, int64_t index);
bool        isnaDouble      (State const&, Value const&, int64_t index);
bool        isnaCharacter   (State const&, Value const&, int64_t index);
bool        isnaPointer     (State const&, Value const&, int64_t index);

// Type checks
bool        isNull          (State const&, Value const&);
bool        isRaw           (State const&, Value const&);
bool        isLogical       (State const&, Value const&);
bool        isInteger       (State const&, Value const&);
bool        isDouble        (State const&, Value const&);
bool        isCharacter     (State const&, Value const&);
bool        isPointer       (State const&, Value const&);
bool        isList          (State const&, Value const&);
bool        isEnvironment   (State const&, Value const&);
bool        isClosure       (State const&, Value const&);

// Value constructors
Value       Null            (State const&, int64_t length);
Value       Raw             (State const&, int64_t length);
Value       Logical         (State const&, int64_t length);
Value       Integer         (State const&, int64_t length);
Value       Double          (State const&, int64_t length);
Value       Character       (State const&, int64_t length);
Value       Pointer         (State const&, int64_t length, Finalizer finalizer);
Value       List            (State const&, int64_t length);
Value       Environment     (State const&, Value const& scope);
Value       Closure         (State const&, Value const& args, Value const& expr, Value const& scope);

Value		newVector       (State const&, Type type, int64_t length);

// Value properties 
Type        typeOf          (State const&, Value const&);
int64_t     length          (State const&, Value const&);

Value       getSlot         (State const&, Value const&, String slot);
Value       setSlot         (State const&,
                             Value const&,
                             String slot,
                             Value const& update)
Value       slots           (State const&, Value const&);
// Duplicate another value's slots?
// Or have a special slot data structure that is publically visible?
// Or store it as a list and make lists a richer structure.
Value       strip           (State const&, Value const&);

Value       get             (State const&, Value const&, Value const& index);
Value       set             (State const&,
                             Value const&,
                             Value const& index,
                             Value const& update);
Value       getSlice        (State const&, Value const&, Value const& index);
Value       setSlice        (State const&,
                             Value const&,
                             Value const& index,
                             Value const& update);
Value       cast            (State const&, Value const&, Type type);
bool        isNA            (State const&, Value const&, Value const& index);

Value       getScope        (State const&, Value const&);
Value       setScope        (State const&, Value const&, Value const& env); 


// Access to special values
Value       globalEnvironment (State const&);


// Language
Value       eval            (State const&, Value const& expr, Value const& context);
Value       parse           (State const&, String s);
Value       call            (State const&, Value const& fn, Value const& args); // same as get

// Environment specials
bool        define          (State const&, Value const& env, Value const&);
bool        undefine        (State const&, Value const& env, Value const&);

Value       names           (State const&, Value const& env);

void        setPromise      (State const&,
                             Value const& env,
                             Value const& index,
                             Value const& expr);
bool        isPromise       (State const&, Value const& env, Value const& index);
Value       getPromiseExpr  (State const&, Value const& env, Value const& index);
Value       getPromiseEnv   (State const&, Value const& env, Value const& index);

Value       getActive       (State const&, Value const& env, Value const& index);
void        setActive       (State const&,
                             Value const& env,
                             Value const& index,
                             Value const& active);
bool        isActive        (State const&, Value const& env, Value const& index);

void        lock            (State const&, Value const& env);
void        unlock          (State const&, Value const& env);
void        isLocked        (State const&, Value const& env);

void        lockBinding     (State const&, Value const& env, Value const& index);
void        unlockBinding   (State const&, Value const& env, Value const& index);
void        isLockedBinding (State const&, Value const& env, Value const& index);

// Closure specials
Value       getBody         (State const&, Value const& f);
Value       setBody         (State const&, Value const& f, Value const&);

Value       getFormals      (State const&, Value const& f);
Value       setFormals      (State const&, Value const& f, Value const&);

// Operators
bool        equal           (State const&, Value const& a, Value const& b);

// GC access
void        pushGC          (State const&, Value const&);
void        popGC           (State const&, Value const&);
// GC stack region? Maybe pass in a function ptr? 

void        frame           (State const&, int64_t index);
*/

// Type for C functions called via .Riposte
// Users should not try to access arguments outside the range [0..argc-1].
//typedef Value (*dotRiposte)(State&, Value* argv);



/*
	Set NAs in a vector

	Populate a vector efficiently
	(Lists could be clojure style structures!)
	(They could all be now that I have a way to coerce them to flat structures
	  when in a C call.)
	(Once flattened for a C call, C code cannot assign past the end)

    MACROS
	Recursively find a symbol in an environment chain
	Recursively find a symbol of a particular type (function) in an environment chain

	Missing

	(Should make a "Promise" an R API concept where it has both the value and the expression and the environment.)
	(With a bit of effort, could even delay evaluation until R actually wants it evaluated.)

	define .Riposte API
	and .Map, .Fold, and .Scan APIs

	errors and warnings	

	Futures should be completely invisible at the Riposte API level.
	As a first attempt, we can just materialize futures at the API boundaries.
	Longer term, we should be able to propagate them through
		API calls where possible.

	Vector generators
	
	Promises should only exist in an environment. They shouldn't be a "real"
		value type.
	Nil should only exist in an environment where it represents an undefined
		value.

	Visible/invisible?
*/
/*
// Helpers
Value        getClass       (State const&, Value const&);
Value        getNames       (State const&, Value const&);
Value        newSymbol      (State const&, Value const&);
Value        newExpression  (State const&, Value const&);
Value        newCall        (State const&, Value const&);
Value        newPairlist    (State const&, Value const&);

bool        isSymbol        (State const&, Value const&);
bool        isCall          (State const&, Value const&);
bool        isExpression    (State const&, Value const&);
bool        isPairlist      (State const&, Value const&);
*/

/*inline
Value       Integer1     (State const& state, int64_t i) {
    Value r = Integer(state, 1);
    setInteger(state, r, 0, i);
    return r;
}*/

}

#endif

