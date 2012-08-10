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
bool Guard_Type(Thread& thread, int64_t i, Type::Enum t) {
    OPERAND(a, i);
    return a.type == t;
}

extern "C"
bool Guard_TypeWidth(Thread& thread, int64_t i, Type::Enum t, size_t width) {
    OPERAND(a, i);
    return a.type == t && a.length == width;
}

extern "C"
void const* Load_double(Thread& thread, int64_t i) {
    OPERAND(a, i);
    return ((Double const&)a).v();
}

extern "C"
void const* Load_integer(Thread& thread, int64_t i) {
    OPERAND(a, i);
    return ((Integer const&)a).v();
}

extern "C"
void const* Load_logical(Thread& thread, int64_t i) {
    OPERAND(a, i);
    return ((Logical const&)a).v();
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

