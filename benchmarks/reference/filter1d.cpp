#include <stdio.h>
int main() {
	int N = 10;
	double * a = new double[N];
	double * r = new double[N];
	for(int i = 0; i <  N; i++) {
		a[i] = i+1;
		r[i] = 0;
	}
	
	double f[] = { -.5, -1,3,-1,-0.5 };
	
	for(int i = 1; i <= 5; i++) {
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
	}
}