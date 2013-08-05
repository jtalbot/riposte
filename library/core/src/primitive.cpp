
#include <math.h>
#include <cstdio>

#include "../../../src/compiler.h"
#include "../../../src/parser.h"

extern "C"
void nchar_map(Thread& thread, int64_t& r, String s) {
    r = strlen(s);
}

extern "C"
void nzchar_map(Thread& thread, Logical::Element& r, String s) {
    r = (*s != 0) ? Logical::TrueElement 
                  : Logical::FalseElement;
}


extern "C"
void isnan_map(Thread& thread, Logical::Element& r, double a) {
    r = (a != a) ? Logical::TrueElement
                 : Logical::FalseElement;
}

extern "C"
void iabs_map(Thread& thread, int64_t& r, int64_t a) {
    r = Integer::isNA(a) ? Integer::NAelement : abs(a);
}

extern "C"
void fabs_map(Thread& thread, double& r, double a) {
    r = fabs(a);
}

extern "C"
void sqrt_map(Thread& thread, double& r, double a) {
    r = sqrt(a);
}

extern "C"
void sign_map(Thread& thread, double& r, double a) {
    r = (a>0)-(a<0);
}

extern "C"
void floor_map(Thread& thread, double& r, double a) {
    r = floor(a);
}

extern "C"
void ceiling_map(Thread& thread, double& r, double a) {
    r = ceil(a);
}

extern "C"
void trunc_map(Thread& thread, double& r, double a) {
    r = trunc(a);
}

extern "C"
void log_map(Thread& thread, double& r, double a) {
    r = log(a);
}

extern "C"
void exp_map(Thread& thread, double& r, double a) {
    r = exp(a);
}

extern "C"
void cos_map(Thread& thread, double& r, double a) {
    r = cos(a);
}

extern "C"
void sin_map(Thread& thread, double& r, double a) {
    r = sin(a);
}

extern "C"
void tan_map(Thread& thread, double& r, double a) {
    r = tan(a);
}

extern "C"
void acos_map(Thread& thread, double& r, double a) {
    r = acos(a);
}

extern "C"
void asin_map(Thread& thread, double& r, double a) {
    r = asin(a);
}

extern "C"
void atan_map(Thread& thread, double& r, double a) {
    r = atan(a);
}

extern "C"
void atan2_map(Thread& thread, double& r, double a, double b) {
    r = atan2(a,b);
}

extern "C"
void cosh_map(Thread& thread, double& r, double a) {
    r = cosh(a);
}

extern "C"
void sinh_map(Thread& thread, double& r, double a) {
    r = sinh(a);
}

extern "C"
void tanh_map(Thread& thread, double& r, double a) {
    r = tanh(a);
}

extern "C"
void acosh_map(Thread& thread, double& r, double a) {
    r = acosh(a);
}

extern "C"
void asinh_map(Thread& thread, double& r, double a) {
    r = asinh(a);
}

extern "C"
void atanh_map(Thread& thread, double& r, double a) {
    r = atanh(a);
}

extern "C"
void hypot_map(Thread& thread, double& r, double a, double b) {
    r = hypot(a,b);
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
void round_map(Thread& thread, double& r, double a, int64_t b) {
    r = riposte_round(thread, a, b);
}

extern "C"
void signif_map(Thread& thread, double& r, double a, int64_t b) {
    r = riposte_signif(thread, a, b);
}

extern "C"
void concat_map(Thread& thread, String& r, String a, String b) {
    r = thread.internStr(std::string(a) + b);
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
void concat_fini(Thread& thread, void* state, String& r) {
    r = thread.internStr(*((std::string*)state));
    delete (std::string*)state;
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
void escape_map(Thread& thread, String& r, String a) {
    r = thread.internStr( escape(thread.externStr(a)) );
}

