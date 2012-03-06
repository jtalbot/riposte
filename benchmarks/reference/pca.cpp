
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <algorithm>

#include "Dense"

using Eigen::MatrixXd;
using Eigen::Map;

#include<iostream>

#include "timing.h"

#define R 100000
#define C 50

//double results[2];
int main() {

	double (*data)[R] = (double (*) [R]) malloc(R * C * sizeof(double));
	FILE * file = fopen("../data/pca.txt","r");
	assert(file);
	
	for(int c = 0; c < C; c++) {
		for(int r = 0; r < R; r++) {
			fscanf(file,"%lf",&data[c][r]);
		}
	}
	
	fclose(file);
	
	double begin = current_time();

	double * means = new double[C];
	std::fill(means,means+C,0.0);
	for(int c = 0; c < C; c++) {
		double m = 0.0;
		for(int r = 0; r < R; r++) {
			m += (data[c][r] - m)/(r+1);
		}
		means[c] = m;
	}
	//for(int c = 0; c < C; c++)
	//  printf("%f\n",means[c]);
	
	double (*cov)[C] = (double (*) [C]) malloc(C * C * sizeof(double));
	bzero(cov,sizeof(double) * C * C);
	
	
	#define BLOCK_SIZE 16384
	for(int rr = 0; rr < R; rr += BLOCK_SIZE) {
		for(int c0 = 0; c0 < C; c0++) {
			for(int c1 = c0; c1 < C; c1++) {
				
				__m128d b = _mm_set_pd(means[c0],means[c0]);
				__m128d d = _mm_set_pd(means[c1],means[c1]);
					
				__m128d acc = _mm_setzero_pd();
				int end = std::min(R,rr + BLOCK_SIZE);
				for(int r = rr; r < end; r += 2) {
					__m128d a = _mm_load_pd(&data[c0][r]);
					__m128d c = _mm_load_pd(&data[c1][r]);
					__m128d r2 = _mm_mul_pd(_mm_sub_pd(a,b),_mm_sub_pd(c,d));
					
					acc = _mm_add_pd(acc,r2);
					
					//v += (data[c0][r] - means[c0]) * (data[c1][r] - means[c1]);
				}
				union {
					__m128d acc2;
					double results[2];
				};
				acc2 = acc;
				//_mm_store_pd(results,acc);
				double v  = results[0] + results[1];
				double cv =  v / (R - 1);
				cov[c0][c1] += cv;
				if(c0 != c1)
					cov[c1][c0] += cv;
			}
		}
	}
	
	printf("cov %f\n",current_time() - begin);
	
	MatrixXd cov_m = Map<MatrixXd>((double*)cov, C, C);
	
	
	//std::cout << cov_m << std::endl;
	
	double (*eig)[C] = (double (*)[C]) malloc(C * C * sizeof(double));
	
	MatrixXd eig_m  = Map<MatrixXd>((double*)eig, C, C);
	
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(cov_m);
	printf("begin\n");
	eig_m = eigensolver.eigenvectors();
	
	double end = current_time();
	
	std::cout << eig_m(0,0) << std::endl;
	printf("Elapsed: %f\n",end - begin);
	//std::cout << eig_m << std::endl;
/*
	double (*result)[C] = (double (*)[C]) malloc(R * C * sizeof(double));
	
	
	
	MatrixXd data_m = Map<MatrixXd>((double*)data, R, C);
	Map<MatrixXd>((double*)result, R, C) = data_m * eig_m;	
*/	
	return 0;
}
