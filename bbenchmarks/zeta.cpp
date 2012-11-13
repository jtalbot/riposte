
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>

#include "../src/common.h"

double* zeta(int64_t M, int64_t N) {
   
    double* r = new double[M];
    for(int i = 0; i < M; i++)
        r[i] = 0;

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            r[i] += pow(j, -(3+i));
        }
    }
 
    return r;
}

int main(int argc, char** argv) {
    static const int64_t M = MM;
    static const int64_t N = NN;

    timespec b = get_time();
    volatile double* r = zeta(M, N/M);
    printf("%f", time_elapsed(b));
}
