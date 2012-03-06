#include <stdio.h>

#include "timing.h"

int main() {
	int N = 10000000;
	double * a = new double[N];
	double * r = new double[N];
	for(int i = 0; i <  N; i++) {
		a[i] = i+1;
		r[i] = 0;
	}
	
	double f[] = { 0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1 };
	
	/*for(int i = 1; i <= 5; i++) {
		int cur = 0;
		for(int j = 0; j < N; j++) {
			if(j >= (i - 1) && (j < (N - 5) || j >= (N - 5) + (5 - i))) {
				r[cur++] += a[j] * f[i - 1];
			}
		}
		//for(int j = 0; j < N; j++) {
		//	printf("%f ",r[j]);
		//}
		//printf("\n");
	}*/

	double begin = current_time();

	for(int i = 0; i < N-7; i++) {
		r[i] = f[0]*a[i] 
			+ f[1]*a[i+1]
			+ f[2]*a[i+2]
			+ f[3]*a[i+3]
			+ f[4]*a[i+4]
			+ f[5]*a[i+5]
			+ f[6]*a[i+6];
	}

	printf("%f\n", current_time()-begin);
}
