
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "timing.h"

double drand() {
	double s1 = rand()  /  (double) RAND_MAX;
	return s1;
}

int main() {
	static const int N = 1000000;
	static const int M = 10000000;
	double result[N];

	double* v = new double[N];
	int* row_idx = new int[M];
	int* col_idx =new  int[M];
	double* values = new double[M];

	for(int i = 0; i < N; i++) 
		result[i] = 0;
	
	for(int i = 0; i < M; i++) {
		row_idx[i] = (int)(drand()*N);
		col_idx[i] = (int)(drand()*N);
		values[i] = drand();
	}
	std::sort(&col_idx[0], &col_idx[M]);

	double begin = current_time();

	/*for(int r = 0; r < M; r++) {
		double a = 0;
		for(int i = row_idx[r]; i < row_idx[r+1]; i++) {
			int c = columns[i];
			a += values[i] * v[c]; 
		}
		result[r] = a;
		
	}*/

	for(int i = 0; i < M; i++) {
		int c = col_idx[i];
		int r = row_idx[i];
		result[r] += values[i] * v[c];
	}

	printf("Elapsed: %f\n", current_time()-begin);
	printf("%f %f %f %f\n",result[0],result[1],result[2],result[3]);
}
