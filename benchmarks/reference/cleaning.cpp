#include <cmath>
#include <stdio.h>
#include "timing.h"

#include <xmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
//#include <immintrin.h>

bool is_outlier(double x) { return x == 9999; }


#define VECTORIZE

double s_mean[2];
double s_mean_2[2];
double s_N_filtered[2];

int main() {
	const int N = 20000000;
	double * data = new double[N];
	
	for(int i = 0; i < N; i++)
		data[i] = i + 1;

	double begin = current_time();		
	
	
	double r = 0.0;
	
#ifndef VECTORIZE
	double mean = 0.0;
	double mean_2 = 0.0;
	
	int N_filtered = 0;
#define INCR 1
#else
	__m128d g_mean = _mm_setzero_pd();
	__m128d g_mean_2 = _mm_setzero_pd();
	__m128d g_N_filtered = _mm_setzero_pd();
	
	__m128d one = _mm_set_pd(1.0,1.0);
	__m128d nines = _mm_set_pd(9999,9999);
#define INCR 2
#endif

	for(int i = 0; i < N; i += INCR) {
	
#ifdef VECTORIZE
		__m128d N_filtered = _mm_add_pd(g_N_filtered,one);
		__m128d data_i = _mm_load_pd(&data[i]);
		
		__m128d mean = _mm_add_pd(g_mean,_mm_div_pd(_mm_sub_pd(data_i, g_mean), N_filtered));
		__m128d mean_2 = _mm_add_pd(g_mean_2,_mm_mul_pd(data_i,data_i));
		
		__m128d mask = _mm_cmpeq_pd(nines,data_i);
		
		g_mean = _mm_blendv_pd(mean,g_mean,mask);
		g_mean_2 = _mm_blendv_pd(mean_2,g_mean_2,mask);
		g_N_filtered = _mm_blendv_pd(N_filtered,g_N_filtered,mask);
		
		
#else
		if(!is_outlier(data[i])) {
			mean += (data[i] - mean) / ++N_filtered;
			mean_2 += data[i] * data[i];
		}
#endif
	}
	
	
#ifdef VECTORIZE
	
	_mm_store_pd(s_mean,g_mean);
	_mm_store_pd(s_mean_2,g_mean_2);
	_mm_store_pd(s_N_filtered,g_N_filtered);
	double mean = (s_mean[0] + s_mean[1]) / 2.0;
	double mean_2 = (s_mean_2[0] + s_mean_2[1]);
	double N_filtered = (s_N_filtered[0] + s_N_filtered[1]);
	
#endif
	double stddev = sqrt( (mean_2 / N_filtered  - mean * mean) * N_filtered / (N_filtered - 1));


	//printf("%f %f\n", mean,stddev);
	
	for(int i = 0; i < N; i++) {
		if(!is_outlier(data[i]))
			r += fabs((data[i] - mean) / stddev) > 1;
	}
	
	printf("Result: %f\n",r);
	printf("Elapsed: %f\n", current_time()-begin);
}
