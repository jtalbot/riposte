
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include "timing.h"

double drand() {
	double s1 = rand()  /  (double) RAND_MAX;
	return s1;
}

extern "C" {
	void smv_loop(int M, int64_t col_idx[], int64_t row_idx[], double values[],double v[],double result[]);
}
int main() {
	static const int N = 500000;
	static const int M = 20000000;
	double result[N];

	double* v = new double[N];
	int64_t* row_idx = new int64_t[M];
	int64_t* col_idx =new  int64_t[M];
	double* values = new double[M];

	for(int i = 0; i < N; i++) { 
		result[i] = 0;
		v[i] = drand();
	}
	
	for(int i = 0; i < M; i++)
		row_idx[i] = (int)(drand()*N);
	for(int i = 0; i < M; i++)
		col_idx[i] = (int)(drand()*N);
	for(int i = 0; i < M; i++)
		values[i] = drand();
	std::sort(&row_idx[0], &row_idx[M]);

	double begin = current_time();

#if 0
	smv_loop(M,col_idx,row_idx,values,v,result);
#else
	for(int i = 0; i < M; i++) {
		int c = col_idx[i];
		int r = row_idx[i];
		result[r] += values[i] * v[c];
	}
#endif
	printf("Elapsed: %f\n", current_time()-begin);
	//printf("%f %f %f %f\n",result[0],result[1],result[2],result[3]);
}
