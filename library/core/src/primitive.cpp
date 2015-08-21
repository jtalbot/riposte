
#include <math.h>
#include <cstdio>

#include "../../../src/compiler.h"
#include "../../../src/parser.h"

extern "C"
void nchar_map(State& state, int64_t& r, String s) {
    r = strlen(s->s);
}

extern "C"
void nzchar_map(State& state, Logical::Element& r, String s) {
    r = (s->s[0] != 0) ? Logical::TrueElement 
                       : Logical::FalseElement;
}


extern "C"
void isnan_map(State& state, Logical::Element& r, double a) {
    r = (a != a) ? Logical::TrueElement
                 : Logical::FalseElement;
}

extern "C"
void iabs_map(State& state, int64_t& r, int64_t a) {
    r = Integer::isNA(a) ? Integer::NAelement : abs(a);
}

extern "C"
void fabs_map(State& state, double& r, double a) {
    r = fabs(a);
}

extern "C"
void sqrt_map(State& state, double& r, double a) {
    r = sqrt(a);
}

extern "C"
void sign_map(State& state, double& r, double a) {
    r = (a>0)-(a<0);
}

extern "C"
void floor_map(State& state, double& r, double a) {
    r = floor(a);
}

extern "C"
void ceiling_map(State& state, double& r, double a) {
    r = ceil(a);
}

extern "C"
void trunc_map(State& state, double& r, double a) {
    r = trunc(a);
}

extern "C"
void log_map(State& state, double& r, double a) {
    r = log(a);
}

extern "C"
void exp_map(State& state, double& r, double a) {
    r = exp(a);
}

extern "C"
void cos_map(State& state, double& r, double a) {
    r = cos(a);
}

extern "C"
void sin_map(State& state, double& r, double a) {
    r = sin(a);
}

extern "C"
void tan_map(State& state, double& r, double a) {
    r = tan(a);
}

extern "C"
void acos_map(State& state, double& r, double a) {
    r = acos(a);
}

extern "C"
void asin_map(State& state, double& r, double a) {
    r = asin(a);
}

extern "C"
void atan_map(State& state, double& r, double a) {
    r = atan(a);
}

extern "C"
void atan2_map(State& state, double& r, double a, double b) {
    r = atan2(a,b);
}

extern "C"
void cosh_map(State& state, double& r, double a) {
    r = cosh(a);
}

extern "C"
void sinh_map(State& state, double& r, double a) {
    r = sinh(a);
}

extern "C"
void tanh_map(State& state, double& r, double a) {
    r = tanh(a);
}

extern "C"
void acosh_map(State& state, double& r, double a) {
    r = acosh(a);
}

extern "C"
void asinh_map(State& state, double& r, double a) {
    r = asinh(a);
}

extern "C"
void atanh_map(State& state, double& r, double a) {
    r = atanh(a);
}

extern "C"
void hypot_map(State& state, double& r, double a, double b) {
    r = hypot(a,b);
}

inline double riposte_round(State& state, double a, int64_t b) { 
    double s = pow(10, b);
    double i = floor(a*s);
    double r = a*s - i;
    if(r > 0.5) return (i+1)/s;
    else if(r < 0.5) return i/s;
    else if(i/2 - floor(i/2) < 0.25) return i/s;
    else return (i+1)/s;
}

inline double riposte_signif(State& state, double a, int64_t b) {
	double d = ceil(log10(a < 0 ? -a : a));
	return riposte_round(state, a,b-(int64_t)d);
}

extern "C"
void round_map(State& state, double& r, double a, int64_t b) {
    r = riposte_round(state, a, b);
}

extern "C"
void signif_map(State& state, double& r, double a, int64_t b) {
    r = riposte_signif(state, a, b);
}

extern "C"
void concat_map(State& state, String& r, String a, String b) {
    r = MakeString(std::string(a->s) + b->s);
}

extern "C"
void* concat_init(State& state) {
    return (void*)(new std::string(""));
}

extern "C"
void concat_op(State& state, void* accumulator, String a) {
    *((std::string*)accumulator) += a->s;
}

extern "C"
void concat_fini(State& state, void* accumulator, String& r) {
    r = MakeString(*((std::string*)accumulator));
    delete (std::string*)accumulator;
}

struct mean_accumulator {
    double k;
    double m;
};

extern "C"
void* mean_init(State& state) {
    mean_accumulator* s = new mean_accumulator();
    s->k = 0;
    s->m = 0;
    return (void*)s;
}

extern "C"
void mean_op(State& state, void* accumulator, double d) {
    mean_accumulator* s = (mean_accumulator*)accumulator;
    s->k += 1.0;
    s->m += (d-(s->m))/s->k;
}

extern "C"
double mean_fini(State& state, void* accumulator) {
    double r = ((mean_accumulator*)accumulator)->m;
    delete (mean_accumulator*)accumulator;
    return r;
}

extern "C"
void escape_map(State& state, String& r, String a) {
    r = MakeString( escape(state.externStr(a)) );
}

