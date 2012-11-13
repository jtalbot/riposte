#include<stdio.h>
#include<math.h>
#include<algorithm>

#include "../src/common.h"

static const double invSqrt2Pi = 0.39894228040;

double cnd(double X) {
	
	double k = 1.0 / (1.0 + 0.2316419 * fabs(X)); 
	double w = (((((1.330274429*k) - 1.821255978)*k + 1.781477937)*k - 0.356563782)*k + 0.31938153)*k;
	w = w * invSqrt2Pi * exp(X * X * -.5);
	if(X > 0) return 1.0-w;
	else return w;
}

double black_scholes(int64_t M, int64_t N, double* X, double* TT, double* r, double* v) {
	double* S = new double[M];

    for(int k = 0; k < M; k++) {
        S[k] = k;
    }

    double sum = 0;
    for(int j = 0; j < N; j++) {

        for(int k = 0; k < M; k++)
            S[k]++;

		for(int i = 0; i < M; i++) {
	
			double delta = v[i] * sqrt(TT[i]);
			double d1 = (log(S[i]/X[i])/log(10) + (r[i] + v[i] * v[i] * .5) * TT[i]) / delta;
			double d2 = d1 - delta;
			sum += S[i] * cnd(d1) - X[i] * exp(-r[i] * TT[i]) * cnd(d2);
		}
	}
    return sum;
}

double * fill(int n, double v) {
	double * a = new double[n];
	std::fill(a,a+n,v);
	return a;
}

int main(int argc, char** argv) {
    static const int64_t M = MM;
    static const int64_t N = NN;

	double * S = fill(M, 100);
	double * X = fill(M, 98);
	double * TT = fill(M, 2);
	double * r = fill(M, .02);
	double * v = fill(M, 5);
	
    timespec b = get_time();
    volatile double d;
    d = black_scholes(M, N/M, X, TT, r, v);
    printf("%f", time_elapsed(b));
}
