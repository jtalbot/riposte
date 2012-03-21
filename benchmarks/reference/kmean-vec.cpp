
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<float.h>
#include <algorithm>

#define N_DIMS 2
#define N_MEANS 5
#define N_POINTS 1000000
#define MAX_ITERATIONS 100
#ifdef VDB_DEBUG
#include <vdb.h>
#endif
#include<assert.h>

#include "timing.h"

/*double drand() {
	double s1 = arc4random()  /  (double) 0xFFFFFFFF;
	return (arc4random() % 2) + .5 * s1;
}*/

extern "C" {
	void kmeans_loop( double * points,  int * count,  double * means,  double * new_means);
}

int main() {
	
	double * points = new double[N_POINTS * N_DIMS];
	double * means = new double [N_MEANS * N_DIMS * 4];
	double * new_means = new double [N_MEANS * N_DIMS * 4];
	int * count = new int[N_MEANS * 4];
	
	FILE * file = fopen("../data/kmeans.txt","r");
	assert(file);
	for(int j = 0; j < N_DIMS * N_POINTS; j++) {
		fscanf(file,"%lf",&points[j]);
	}
	
	for(int i = 0; i < N_MEANS; i++) {
		for(int d = 0; d < N_DIMS; d++) {
			for(int k = 0; k < 4; k++)
				means[N_MEANS * N_DIMS * k + d * N_MEANS + i] = points[d * N_POINTS + i];
		}
	}
	
	double begin = current_time();
	
	for(int n = 0; n < MAX_ITERATIONS; n++) {
		bzero(new_means,sizeof(double) * N_MEANS * N_DIMS * 4);
		bzero(count,sizeof(int) * N_MEANS * N_DIMS * 4);
		
		kmeans_loop((double *) points, (int *) count, (double *) means, (double *) new_means);		
		for(int j = 0; j < N_MEANS *N_DIMS; j++) {
			for(int i = 1; i < 4; i++) {
				new_means[j] += new_means[i * N_DIMS * N_MEANS + j];
			}
			new_means[j] /= 4;
		}
		
		
		std::swap(means,new_means);
	}

	printf("%f\n", current_time()-begin);
	
	/*for(int i = 0; i < N_MEANS; i++) {
		for(int m = 0; m < N_DIMS; m++) {
			printf("%f \n",means[i][m]);
		}
		printf("\n");
	}*/
	
	return 0;
}
