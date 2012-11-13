
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>

#include "../src/common.h"

double* colsum(double* a, int64_t M, int64_t N) {
   
    double* r = new double[M];
    for(int i = 0; i < M; i++)
        r[i] = 0;

    for(int j = 0; j < N; j++) {
        for(int i = 0; i < M; i++) {
            r[i] += a[j*M+i];
        }
    }
 
    return r;
}

int main(int argc, char** argv) {
    static const int64_t M = MM;
    static const int64_t N = NN;

    double* a = new double[N/M*M];
    for(int i = 0; i < N/M*M; i++)
        a[i] = (double)rand()/RAND_MAX;

    timespec b = get_time();
    volatile double* r = colsum(a, M, N/M);
    printf("%f", time_elapsed(b));
}
