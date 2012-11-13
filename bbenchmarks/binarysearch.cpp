
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>

#include "../src/common.h"

static const int64_t K = 10000000;

int64_t* binary_search(double* v, double* key, int64_t M) {
    int64_t* a = new int64_t[M];
    int64_t* b = new int64_t[M];
    
    for(int i = 0; i < M; i++) {
        a[i] = 0;
        b[i] = K-1;
    }

    for(int i = 0; i < M; i++) {
        while(a[i] < b[i]) {
            int64_t t = (a[i]+b[i]) / 2;
            if(v[t] < key[i])
                a[i] = t+1;
            else
                b[i] = t;
        }
    }

    return a;
}

volatile int64_t* run(double* v, int64_t M, int64_t N) {
    volatile int64_t* r;
    for(int64_t i = 0; i < N; i++) {
        double* key = new double[M];
        for(int64_t j = 0; j < M; j++) {
            key[j] = i*(K/N)+j;
        }
        r = binary_search(v, key, M); 
    }
    return r;
}

int main(int argc, char** argv) {
    static const int64_t M = MM;
    static const int64_t N = NN;

    double* v = new double[K];
    for(int i = 0; i < K; i++)
        v[i] = i+1;

    timespec b = get_time();
    volatile int64_t* r = run(v, M, N/M);
    printf("%f", time_elapsed(b));
}
