
#include "api.h"
#include <R_ext/Random.h>

void GetRNGstate(void) {
    printf("GetRNGstate called, but it doesn't do anything yet.\n");
    //_NYI("GetRNGstate");
}

void PutRNGstate(void) {
    printf("PutRNGstate called, but it doesn't do anything yet.\n");
    //_NYI("PutRNGstate");
}

// Normalizes the probability distribution
// TODO: R also checks for some error conditions
void FixupProb(double* p, int n, int, Rboolean) {
    double sum = 0;
    for(int64_t i = 0; i < n; ++i) {
        sum += p[i];
    }
    for(int64_t i = 0; i < n; ++i) {
        p[i] /= sum;
    }
}

