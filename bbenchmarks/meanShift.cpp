
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>

#include "../src/common.h"

double meanshift(double x, double* X, int64_t M) {
    
    double top = 0, bottom = 0;

    for(int64_t i = 0; i < M; i++) {
        double d = x-X[i];
        double t = exp(-d*d);
        top += t * X[i];
        bottom += t;
    }
    return top/bottom;
}

double run(double* X, int64_t M, int64_t N) {
    double x = 0.5;
    for(int64_t i = 0; i < N; i++) {
        x = meanshift(x, X, M);
    }
    return x;
} 

int main(int argc, char** argv) {
    static const int64_t M = MM;
    static const int64_t N = NN;

    double* a = new double[M];
    for(int64_t i = 0; i < M; i++)
        a[i] = (double)rand() / RAND_MAX;

    timespec b = get_time();
    volatile double r = run(a, M, N/M);
    //printf("%f\n", r);
    printf("%f", time_elapsed(b));
}
