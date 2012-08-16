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
double const* Load_double(Thread& thread, int64_t i) {
    OPERAND(a, i);
    return ((Double const&)a).v();
}

extern "C"
int64_t const* Load_integer(Thread& thread, int64_t i) {
    OPERAND(a, i);
    return ((Integer const&)a).v();
}

extern "C"
char const* Load_logical(Thread& thread, int64_t i) {
    OPERAND(a, i);
    return ((Logical const&)a).v();
}

extern "C"
Prototype* Load_function(Thread& thread, int64_t i) {
    OPERAND(a, i);
    return ((Function const&)a).prototype();
}

extern "C"
void Store_double(Thread& thread, int64_t i, size_t len, double* d) {
	Double a(len);
	memcpy(a.v(), d, len*sizeof(double));
    if(i <= 0) thread.base[i] = a;
    else thread.frame.environment->insertRecursive((String)(i)) = a;
}

extern "C"
void Store_integer(Thread& thread, int64_t i, size_t len, int64_t* d) {
	Integer a(len);
	memcpy(a.v(), d, len*sizeof(int64_t));
    if(i <= 0) thread.base[i] = a;
    else thread.frame.environment->insertRecursive((String)(i)) = a;
}

extern "C"
void Store_logical(Thread& thread, int64_t i, size_t len, int8_t* l) {
	Logical a(len);
	memcpy(a.v(), l, len*sizeof(int8_t));
    if(i <= 0) thread.base[i] = a;
    else thread.frame.environment->insertRecursive((String)(i)) = a;
}

extern "C"
void Store_function(Thread& thread, int64_t i, size_t len, Prototype* p) {
    printf("Store_function NYI\n");
}

