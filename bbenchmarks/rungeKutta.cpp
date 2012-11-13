
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <stdlib.h>

#include "../src/common.h"

double initial(double v) {
    return sqrt(v);
}

double secondary(double v, double y) {
    return v*y;
}

double* rungeKutta(double* i, double h, int64_t M, int64_t N) {
    double* y = new double[M];
    for(int k = 0; k < M; k++)
        y[k] = initial(i[k]);

    double t = 1;
    while(t < N) {

        for(int k = 0; k < M; k++) {
            double k1 = h * secondary(t, y[k]);
            double k2 = h * secondary(t + 0.5*h, y[k] + 0.5*k1);
            double k3 = h * secondary(t + 0.5*h, y[k] + 0.5*k2);
            double k4 = h * secondary(t + 0.5*h, y[k] + k3);
            y[k] = y[k] + (1.0/6.0)*(k1+k2+k3+k4);
        }
        t = t+1;
    }
    return y;
}

int main(int argc, char** argv) {
    static const int64_t M = MM;
    static const int64_t N = NN;


    double* i = new double[M];
    for(int k = 0; k < M; k++)
        i[k] = k+1;

    timespec b = get_time();
    volatile double* r = rungeKutta(i, 4, M, N/M);
    printf("%f", time_elapsed(b));
}
