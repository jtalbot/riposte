#include<stdio.h>
#include<math.h>
#include<algorithm>
static const int N_OPTIONS = 1024;
static const int N_BLACK_SCHOLES_ROUNDS = 1;

static const double invSqrt2Pi = 0.39894228040;
static const double LOG10 = log(10);

double cnd(double X) {
	
	double k = 1.0 / (1.0 + 0.2316419 * fabs(X)); 
	double w = (((((1.330274429*k) - 1.821255978)*k + 1.781477937)*k - 0.356563782)*k + 0.31938153)*k;
	w = w * invSqrt2Pi * exp(X * X * -.5);
	return w;
	//TODO: if then else
}

double * fill(int n, double v) {
	double * a = new double[n];
	std::fill(a,a+n,v);
	return a;
}

int main() {
	double * S = fill(N_OPTIONS, 100);
	double * X = fill(N_OPTIONS, 98);
	double * TT = fill(N_OPTIONS, 2);
	double * r = fill(N_OPTIONS, .02);
	double * v = fill(N_OPTIONS, 5);
	
	double acc = 0.0;
	for(int j = 0; j < N_BLACK_SCHOLES_ROUNDS; j++) {
	
		for(int i = 0; i < N_OPTIONS; i++) {
			double delta = v[i] * sqrt(TT[i]);
			double d1 = (log(S[i]/X[i])/LOG10 + (r[i] + v[i] * v[i] * .5) * TT[i]) / delta;
			double d2 = d1 - delta;
			acc += S[i] * cnd(d1) - X[i] * exp(-r[i] * TT[i]) * cnd(d2);
		}
	}
	acc /= (N_BLACK_SCHOLES_ROUNDS * N_OPTIONS);
	printf("%f\n",acc);
}