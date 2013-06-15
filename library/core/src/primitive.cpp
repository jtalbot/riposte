
#include <math.h>
#include <cstdio>

#include "../../../src/compiler.h"
#include "../../../src/parser.h"

extern "C"
int64_t nchar_map(Thread& thread, String s) {
    return strlen(s);
}

extern "C"
char nzchar_map(Thread& thread, String s) {
    return *s != 0 ? Logical::TrueElement : Logical::FalseElement;
}


extern "C"
char isnan_map(Thread& thread, double a) {
    return Double::isNA(a) ? Logical::NAelement : 
                (a != a) ? Logical::TrueElement :
                           Logical::FalseElement;
}

extern "C"
int64_t iabs_map(Thread& thread, int64_t a) {
    return Integer::isNA(a) ? Integer::NAelement : abs(a);
}

extern "C"
double fabs_map(Thread& thread, double a) {
    return fabs(a);
}

extern "C"
double sqrt_map(Thread& thread, double a) {
    return sqrt(a);
}

extern "C"
double sign_map(Thread& thread, double a) {
    return (a>0)-(a<0);
}

extern "C"
double floor_map(Thread& thread, double a) {
    return floor(a);
}

extern "C"
double ceiling_map(Thread& thread, double a) {
    return ceil(a);
}

extern "C"
double trunc_map(Thread& thread, double a) {
    return trunc(a);
}

extern "C"
double log_map(Thread& thread, double a) {
    return log(a);
}

extern "C"
double exp_map(Thread& thread, double a) {
    return exp(a);
}

extern "C"
double cos_map(Thread& thread, double a) {
    return cos(a);
}

extern "C"
double sin_map(Thread& thread, double a) {
    return sin(a);
}

extern "C"
double tan_map(Thread& thread, double a) {
    return tan(a);
}

extern "C"
double acos_map(Thread& thread, double a) {
    return acos(a);
}

extern "C"
double asin_map(Thread& thread, double a) {
    return asin(a);
}

extern "C"
double atan_map(Thread& thread, double a) {
    return atan(a);
}

extern "C"
double atan2_map(Thread& thread, double a, double b) {
    return atan2(a,b);
}

extern "C"
double hypot_map(Thread& thread, double a, double b) {
    return hypot(a,b);
}

inline double riposte_round(Thread& thread, double a, int64_t b) { 
    double s = pow(10, b); 
    return round(a*s)/s;
}

inline double riposte_signif(Thread& thread, double a, int64_t b) {
	double d = ceil(log10(a < 0 ? -a : a));
	return riposte_round(thread, a,b-(int64_t)d);
}

extern "C"
double round_map(Thread& thread, double a, int64_t b) {
    return Double::isNA(a) || Integer::isNA(b)
            ? Double::NAelement
            : riposte_round(thread, a, b);
}

extern "C"
double signif_map(Thread& thread, double a, int64_t b) {
    return Double::isNA(a) || Integer::isNA(b)
            ? Double::NAelement
            : riposte_signif(thread, a, b);
}

extern "C"
String concat_map(Thread& thread, String a, String b) {
    return thread.internStr(std::string(a) + b);
}

extern "C"
void* concat_init(Thread& thread) {
    return (void*)(new std::string(""));
}

extern "C"
void concat_op(Thread& thread, void* state, String a) {
    *((std::string*)state) += a;
}

extern "C"
String concat_fini(Thread& thread, void* state) {
    return thread.internStr(*((std::string*)state));
}

struct mean_state {
    double k;
    double m;
};

extern "C"
void* mean_init(Thread& thread) {
    mean_state* s = new mean_state();
    s->k = 0;
    s->m = 0;
    return (void*)s;
}

extern "C"
void mean_op(Thread& thread, void* state, double d) {
    mean_state* s = (mean_state*)state;
    s->k += 1.0;
    s->m += (d-(s->m))/s->k;
}

extern "C"
double mean_fini(Thread& thread, void* state) {
    double r = ((mean_state*)state)->m;
    delete (mean_state*)state;
    return r;
}

extern "C"
String escape_map(Thread& thread, String a) {
    return thread.internStr( escape(thread.externStr(a)) );
}

