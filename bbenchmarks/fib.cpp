
#include <math.h>
#include <stdint.h>
#include <sys/time.h>

#include "../src/common.h"

double* fib(int64_t M, int64_t N) {
    double* a = new double[M];
    double* b = new double[M];
    
    for(int i = 0; i < M; i++) {
        a[i] = 0;
        b[i] = 1;
    }

        for(int j = 0; j < M; j++) {
    for(int i = 0; i < N; i++) {
            double t = b[j];
            b[j] += a[j];
            a[j] = t;
        }
    }

    return a;
}

int main(int argc, char** argv) {
    static const int64_t M = MM;
    static const int64_t N = NN;
    
    timespec b = get_time();
    volatile double* r = fib(M, N/M);
    printf("%f", time_elapsed(b));
}
