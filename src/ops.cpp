#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "interpreter.h"

extern "C"
double loadr_d(Thread& thread, int64_t i) {
	return (thread.base+i)->d;
}

extern "C"
double loadm_d(Thread& thread, int64_t i) {
	Value const& a = thread.frame.environment->getRecursive((String)i);
	return a.d;
}

extern "C"
void storer_d(Thread& thread, int64_t i, double d) {
	Double::InitScalar(thread.base[i], d);
}

extern "C"
void storer_l(Thread& thread, int64_t i, int8_t l) {
	Logical::InitScalar(thread.base[i], l);
}

extern "C"
void storem_d(Thread& thread, int64_t i, double d) {
	Double::InitScalar(
		thread.frame.environment->insert((String)i),
		d);
}

extern "C"
void storem_l(Thread& thread, int64_t i, int8_t l) {
	Logical::InitScalar(
		thread.frame.environment->insert((String)i),
		l);
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

