#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <float.h>

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
		xc[i] = i + 1;
		yc[i] = i + 1;
		zc[i] = i + 1;
	}
	
	double a = 1;
	
	double r = DBL_MAX;
	for(int i = 0; i < n; i++) {
	
		double b = 2*(xd*(xo-xc[i])+yd*(yo-yc[i])+zd*(zo-zc[i]));
		double c = (xo-xc[i])*(xo-xc[i])+(yo-yc[i])*(yo-yc[i])+(zo-zc[i])*(zo-zc[i]);
		
		double t0 = (-b - sqrt(b*b-4*c))/2;
		double t1 = (-b + sqrt(b*b-4*c))/2;
		
		r = std::min(r,std::min(t0,t1));
	}
	printf("%f\n",r);
	return 0;
}