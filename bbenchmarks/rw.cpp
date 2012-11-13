
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>

#include "../src/common.h"

double* rw(int64_t M, int64_t n) {
    double* a = new double[M];
    
    for(int i = 0; i < M; i++) {
        a[i] = 0;
    }

    int64_t i = 0;
    while(i < n) {
        i = i+1;
        for(int j = 0; j < M; j++) {
            a[j] += ((double)rand() / RAND_MAX) < 0.5 ? 1 : -1;
        }
    }

    /*double* a = new double[M];
    
    for(int j = 0; j < M; j++) {
        a[j] = 0;
        int64_t i = 0;
        while(i < n) {
            i = i+1;
            a[j] += ((double)rand() / RAND_MAX) < 0.5 ? 1 : -1;
        }
    }*/

    return a;
}

int main(int argc, char** argv) {
    static const int64_t M = MM;
    static const int64_t N = NN;
    timespec b = get_time();
    volatile double* r = rw(M, N/M);
    printf("%f", time_elapsed(b));
}
