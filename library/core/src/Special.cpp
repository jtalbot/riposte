
#include <math.h>
#include <cstdio>

#include "../../../src/compiler.h"
#include "../../../src/parser.h"

#include "../../../include/Rmath.h"

extern "C"
void beta_map(State& state, double& r, double a, double b) {
    r = beta(a, b);
}

extern "C"
void lbeta_map(State& state, double& r, double a, double b) {
    r = lbeta(a, b);
}

extern "C"
void gamma_map(State& state, double& r, double a) {
    r = gammafn(a);
}

extern "C"
void lgamma_map(State& state, double& r, double a) {
    r = lgammafn(a);
}

extern "C"
void digamma_map(State& state, double& r, double a) {
    r = digamma(a);
}

extern "C"
void trigamma_map(State& state, double& r, double a) {
    r = trigamma(a);
}

extern "C"
void psigamma_map(State& state, double& r, double a, double b) {
    r = psigamma(a, b);
}

extern "C"
void choose_map(State& state, double& r, double a, double b) {
    r = choose(a, b);
}

extern "C"
void lchoose_map(State& state, double& r, double a, double b) {
    r = lchoose(a, b);
}
