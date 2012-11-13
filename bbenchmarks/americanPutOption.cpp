
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>

#include "../src/common.h"

double* americanPut(double T, double* S, double K, double r, double sigma, double q, int64_t M, int64_t N) {

    double deltaT = T/N;
    double up = exp(sigma * sqrt(deltaT));
    double p0 = (up * exp(-r*deltaT) - exp(-q*deltaT)) * up / (up*up-1);
    double p1 = exp(-r * deltaT) - p0;

    double* p = new double[M*N];
    double* rr = new double[M];


    for(int i = 0; i < N; i++) {
        double pp = pow(up, 2*(i+1)-N);
        for(int64_t j = 0; j < M; j++) {
            p[i*M+j] = std::max(0.0, K-S[j] * pp);
        }
    }

    
    for(int k = N-2; k >= 0; k--) {
        for(int i = 0; i <= k; i++) {
            double pp = pow(up, 2*(i+1)-(k+1));
            for(int64_t j = 0; j < M; j++) {
                p[i*M+j] = std::max( K-S[j] * pp,
                                     p0 * p[i*M+j] + p1 * p[(i+1)*M+j] );
            }
        }
    }

    for(int64_t j = 0; j < M; j++) {
        rr[j] = p[j];
    }

    return rr;
}

int main(int argc, char** argv) {
    static const int64_t M = MM;
    static const int64_t N = NN;

    double* S = new double[M];
    for(int i = 0; i < M; i++)
        S[i] = i+1;

    timespec b = get_time();
    volatile double* r = americanPut(100, S, 80, 3, 2, 3, M, ((int)sqrt(N/M)));
    printf("%f", time_elapsed(b));
}
