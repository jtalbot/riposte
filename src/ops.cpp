#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "interpreter.h"

extern "C"
bool loadr_d(Thread& thread, int64_t i, size_t len, double* d) {
    Value const& a = thread.base[i];
	if(a.type != Type::Double) 
        return false;
	memcpy(d, ((Double const&)a).v(), len*sizeof(double));
    return true;
}

extern "C"
bool loadm_d(Thread& thread, int64_t i, size_t len, double* d) {
	Value const& a = thread.frame.environment->getRecursive((String)i);
	if(a.type != Type::Double) 
        return false;
	memcpy(d, ((Double const&)a).v(), len*sizeof(double));
    return true;
}

extern "C"
bool loadr_i(Thread& thread, int64_t i, size_t len, int64_t* d) {
    Value const& a = thread.base[i];
	if(a.type != Type::Integer) 
        return false;
	memcpy(d, ((Integer const&)a).v(), len*sizeof(int64_t));
    return true;
}

extern "C"
bool loadm_i(Thread& thread, int64_t i, size_t len, int64_t* d) {
	Value const& a = thread.frame.environment->getRecursive((String)i);
	if(a.type != Type::Integer) 
        return false;
	memcpy(d, ((Integer const&)a).v(), len*sizeof(int64_t));
    return true;
}

extern "C"
bool loadr_l(Thread& thread, int64_t i, size_t len, int8_t* d) {
    Value const& a = thread.base[i];
	if(a.type != Type::Logical) 
        return false;
	memcpy(d, ((Logical const&)a).v(), len*sizeof(int8_t));
    return true;
}

extern "C"
bool loadm_l(Thread& thread, int64_t i, size_t len, int8_t* d) {
	Value const& a = thread.frame.environment->getRecursive((String)i);
	if(a.type != Type::Logical) 
        return false;
	memcpy(d, ((Logical const&)a).v(), len*sizeof(int8_t));
    return true;
}

extern "C"
void storer_d(Thread& thread, int64_t i, size_t len, double* d) {
	Double a(len);
	memcpy(a.v(), d, len*sizeof(double));
	thread.base[i] = a;
}

extern "C"
void storer_l(Thread& thread, int64_t i, size_t len, int8_t* l) {
	Logical a(len);
	memcpy(a.v(), l, len*sizeof(int8_t));
	thread.base[i] = a;
}

extern "C"
void storem_d(Thread& thread, int64_t i, size_t len, double* d) {
	Double a(len);
	memcpy(a.v(), d, len*sizeof(double));
	thread.frame.environment->insert((String)i) = a;
}

extern "C"
void storem_l(Thread& thread, int64_t i, size_t len, int8_t* l) {
	Logical a(len);
	memcpy(a.v(), l, len*sizeof(int8_t));
	thread.frame.environment->insert((String)i) = a;
}

extern "C"
void store2_ddd(Thread& thread, double* d1, double d2, double d3) {
}

extern "C"
int8_t guard_l(Thread& thread, int8_t v) ALWAYS_INLINE;
extern "C"
int8_t guard_l(Thread& thread, int8_t v) {
	return v;
}

extern "C"
double add_dd(Thread& thread, double a, double b) ALWAYS_INLINE;
extern "C"
double add_dd(Thread& thread, double a, double b) {
	return a+b;
}

extern "C"
int8_t lt_dd(Thread& thread, double a, double b) ALWAYS_INLINE;
extern "C"
int8_t lt_dd(Thread& thread, double a, double b) {
	return a < b ? -1 : 0;
}

