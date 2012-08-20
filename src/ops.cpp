#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"

extern "C"
double const* SLOAD_double(Thread& thread, int64_t i) {
    Value const& a = (thread.registers+DEFAULT_NUM_REGISTERS)[i];
    return a.isDouble() ? ((Double const&)a).v() : 0;
}

extern "C"
int64_t const* SLOAD_integer(Thread& thread, int64_t i) {
    Value const& a = (thread.registers+DEFAULT_NUM_REGISTERS)[i];
    return a.isInteger() ? ((Integer const&)a).v() : 0;
}

extern "C"
char const* SLOAD_logical(Thread& thread, int64_t i) {
    Value const& a = (thread.registers+DEFAULT_NUM_REGISTERS)[i];
    return a.isLogical() ? ((Logical const&)a).v() : 0;
}

extern "C"
Prototype* SLOAD_function(Thread& thread, int64_t i) {
    Value const& a = (thread.registers+DEFAULT_NUM_REGISTERS)[i];
    return a.isFunction() ? ((Function const&)a).prototype() : 0;
}

extern "C"
Prototype* SLOAD_promise(Thread& thread, int64_t i) {
    Value const& a = (thread.registers+DEFAULT_NUM_REGISTERS)[i];
    return a.isFunction() ? ((Function const&)a).prototype() : 0;
}

extern "C"
Prototype* SLOAD_default(Thread& thread, int64_t i) {
    Value const& a = (thread.registers+DEFAULT_NUM_REGISTERS)[i];
    return a.isFunction() ? ((Function const&)a).prototype() : 0;
}

extern "C"
double const* ELOAD_double(Thread& thread, Environment* env, int64_t i) {
    Value const& a = env->getRecursive((String)i);
    return a.isDouble() ? ((Double const&)a).v() : 0;
}

extern "C"
int64_t const* ELOAD_integer(Thread& thread, Environment* env, int64_t i) {
    Value const& a = env->getRecursive((String)i);
    return a.isInteger() ? ((Integer const&)a).v() : 0;
}

extern "C"
char const* ELOAD_logical(Thread& thread, Environment* env, int64_t i) {
    Value const& a = env->getRecursive((String)i);
    return a.isLogical() ? ((Logical const&)a).v() : 0;
}

extern "C"
Prototype* ELOAD_function(Thread& thread, Environment* env, int64_t i) {
    Value const& a = env->getRecursive((String)i);
    return a.isFunction() ? ((Function const&)a).prototype() : 0;
}

extern "C"
Prototype* ELOAD_promise(Thread& thread, Environment* env, int64_t i) {
    Value const& a = env->getRecursive((String)i);
    return a.isFunction() ? ((Function const&)a).prototype() : 0;
}

extern "C"
Prototype* ELOAD_default(Thread& thread, Environment* env, int64_t i) {
    Value const& a = env->getRecursive((String)i);
    return a.isFunction() ? ((Function const&)a).prototype() : 0;
}

extern "C"
Environment* LOAD_environment(Thread& thread, int64_t i) {
    return (Environment*)i;
}

extern "C"
void SSTORE_double(Thread& thread, int64_t i, size_t len, double* d) {
	Double a(len);
	memcpy(a.v(), d, len*sizeof(double));
    (thread.registers+DEFAULT_NUM_REGISTERS)[i] = a;
}

extern "C"
void SSTORE_integer(Thread& thread, int64_t i, size_t len, int64_t* d) {
    Integer a(len);
	memcpy(a.v(), d, len*sizeof(int64_t));
    (thread.registers+DEFAULT_NUM_REGISTERS)[i] = a;
}

extern "C"
void SSTORE_logical(Thread& thread, int64_t i, size_t len, int8_t* d) {
	Logical a(len);
	memcpy(a.v(), d, len*sizeof(int8_t));
    (thread.registers+DEFAULT_NUM_REGISTERS)[i] = a;
}

extern "C"
void SSTORE_function(Thread& thread, int64_t i, size_t len, Prototype* p) {
    Function a;
    Function::Init(a, p, 0);
    (thread.registers+DEFAULT_NUM_REGISTERS)[i] = a;
}

extern "C"
void SSTORE_promise(Thread& thread, int64_t i, size_t len, Prototype* p) {
    Promise a;
    Promise::Init(a, p, 0);
    (thread.registers+DEFAULT_NUM_REGISTERS)[i] = a;
}

extern "C"
void SSTORE_default(Thread& thread, int64_t i, size_t len, Prototype* p) {
    Default a;
    Default::Init(a, p, 0);
    (thread.registers+DEFAULT_NUM_REGISTERS)[i] = a;
}

extern "C"
void SSTORE_NULL(Thread& thread, int64_t i, size_t len, void* p) {
    (thread.registers+DEFAULT_NUM_REGISTERS)[i] = Null::Singleton();
}

extern "C"
void ESTORE_double(Thread& thread, Environment* env, int64_t i, size_t len, double* d) {
	Double a(len);
	memcpy(a.v(), d, len*sizeof(double));
    env->insertRecursive((String)i) = a;
}

extern "C"
void ESTORE_integer(Thread& thread, Environment* env, int64_t i, size_t len, int64_t* d) {
    Integer a(len);
	memcpy(a.v(), d, len*sizeof(int64_t));
    env->insertRecursive((String)i) = a;
}

extern "C"
void ESTORE_logical(Thread& thread, Environment* env, int64_t i, size_t len, int8_t* d) {
	Logical a(len);
	memcpy(a.v(), d, len*sizeof(int8_t));
    env->insertRecursive((String)i) = a;
}

extern "C"
void ESTORE_function(Thread& thread, Environment* env, int64_t i, size_t len, Prototype* p) {
    Function a;
    Function::Init(a, p, 0);
    env->insertRecursive((String)i) = a;
}

extern "C"
void ESTORE_promise(Thread& thread, Environment* env, int64_t i, size_t len, Prototype* p) {
    Promise a;
    Promise::Init(a, p, 0);
    env->insertRecursive((String)i) = a;
}

extern "C"
void ESTORE_default(Thread& thread, Environment* env, int64_t i, size_t len, Prototype* p) {
    Default a;
    Default::Init(a, p, 0);
    env->insertRecursive((String)i) = a;
}

extern "C"
void ESTORE_NULL(Thread& thread, Environment* env, int64_t i, size_t len, void* p) {
    env->insertRecursive((String)i) = Null::Singleton();
}

