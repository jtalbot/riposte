#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <float.h>

#include "timing.h"

extern "C" {
	double raysphere_loop(int n,double start,double,double,double, double xo, double xc[], double yo, double yc[], double zo, double zc[]);
}
int main() {

	int n = 10000000;
	double xo = 0;
	double yo = 0;
	double zo = 0;
	double xd = 1;
	double yd = 0;
	double zd = 0;
	
	double * xc = new double[n];
	double * yc = new double[n];
	double * zc = new double[n];
	for(int i = 0; i < n; i++) {
		xc[i] = i;
		yc[i] = i;
		zc[i] = i;
	}
	
	double begin = current_time();

	double a = 1;
	
	
	#if 0
	double r = raysphere_loop(n,DBL_MAX, xd,yd,zd,xo, xc, yo, yc, zo, zc);
	#else
	
	double r = DBL_MAX;
	for(int i = 0; i < n; i++) {
	
		double b = 2*(xd*(xo-xc[i])+yd*(yo-yc[i])+zd*(zo-zc[i]));
		double c = (xo-xc[i])*(xo-xc[i])+(yo-yc[i])*(yo-yc[i])+(zo-zc[i])*(zo-zc[i])-1;
		
		double disc = b*b-4*c;
		if(disc > 0) {
			double t0 = (-b - sqrt(disc))/2;
			double t1 = (-b + sqrt(disc))/2;
			r = std::min(r,std::min(t0,t1));
		}
	}
	#endif
	printf("%f\n",r);
	printf("Elapsed: %f\n", current_time()-begin);
	return 0;
}
