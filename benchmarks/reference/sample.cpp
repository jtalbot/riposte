
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "timing.h"

double ca[] = {
   	-3.969683028665376e+01,
   	 2.209460984245205e+02,
   	-2.759285104469687e+02,
   	 1.383577518672690e+02,
   	-3.066479806614716e+01,
   	 2.506628277459239e+00 };

double cb[] = {-5.447609879822406e+01,
   	 1.615858368580409e+02,
   	-1.556989798598866e+02,
   	 6.680131188771972e+01,
   	-1.328068155288572e+01 };


double cc[] = {-7.784894002430293e-03,
	-3.223964580411365e-01,
   	-2.400758277161838e+00,
  	-2.549732539343734e+00,
  	 4.374664141464968e+00,
   	 2.938163982698783e+00 };

double cd[] = { 7.784695709041462e-03,
	 3.224671290700398e-01,
   	 2.445134137142996e+00,
   	 3.754408661907416e+00 };

double cdf(double p) { 
	if(p < 0.02425) {
		double q = sqrt(-2*log(p));
		 return (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) /
		 ((((cd[0]*q+cd[1])*q+cd[2])*q+cd[3])*q+1);
	} else if(p <= (1-0.02425)) {
		double q = (p-0.5);
		double r = q*q;
		return (((((ca[0]*r+ca[1])*r+ca[2])*r+ca[3])*r+ca[4])*r+ca[5])*q /
		 (((((cb[0]*r+cb[1])*r+cb[2])*r+cb[3])*r+cb[4])*r+1);
	} else {
		double q = sqrt(-2*log(1-p));
		return -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) /
		  ((((cd[0]*q+cd[1])*q+cd[2])*q+cd[3])*q+1);
	}
}

double means[] = { 0,2,10 };
double sd[] = { 1,0.1,3 };
#define BLK 100
int main() {
	int N = 10000000;
	double * b = new double[N];

	double begin = current_time();
	#if 1
	for(int i = 0; i < N; i += BLK) {
		double a[BLK];
		int idx[BLK];
		for(int j = 0; j < BLK; j++) {
			a[j] = rand() / (double) 0xFFFFFFFF;
			idx[j] = rand() % 3;
		}
		for(int j = 0; j < BLK; j++)
			b[i+j] = cdf(a[j]) * sd[idx[j]] + means[idx[j]];
	}
	#else
	for(int i = 0; i < N; i++) {
		double a = rand() / (double) 0xFFFFFFFF;
		size_t idx = rand() % 3;
		b[i] = cdf(a) * sd[idx] + means[idx];
	}
	#endif
	printf("%f\n", b[0]);
	printf("Elapsed: %f\n", current_time()-begin);
}
