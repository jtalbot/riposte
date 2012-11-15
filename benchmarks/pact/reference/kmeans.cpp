
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
int main() {
	double (*points)[N_DIMS] = (double(*)[N_DIMS]) malloc(sizeof(double) * N_POINTS * N_DIMS);
	
	double (*means)[N_DIMS] = (double(*)[N_DIMS]) malloc(sizeof(double) * N_POINTS * N_DIMS);
	double (*new_means)[N_DIMS] = (double(*)[N_DIMS]) malloc(sizeof(double) * N_POINTS * N_DIMS);
	int * count = new int[N_POINTS];
	
	#ifdef VDB_DEBUG
	vdb_color(1,1,1);
	vdb_point(.75,.75,0);
	#endif
	
	FILE * file = fopen("../data/kmeans.txt","r");
	assert(file);
	for(int j = 0; j < N_DIMS; j++) {
		for(int i = 0; i < N_POINTS; i++) {
			//points[i][j] = drand();
			fscanf(file,"%lf",&points[i][j]);
		}
	}
	
	
	#ifdef VDB_DEBUG
	for(int i = 0; i < N_POINTS; i++) {
		vdb_sample(.001);
		for(int x = 0; x < MAX_ITERATIONS; x++) {
			vdb_point(points[i][0],points[i][1],x);
		}
	}
	#endif
		
	for(int i = 0; i < N_MEANS; i++) {
		for(int j = 0; j < N_DIMS; j++) {
			means[i][j] = points[i][j];
		}
	}
	
	#ifdef VDB_DEBUG
	vdb_sample(1);
	vdb_color(1,0,0);
	for(int i = 0; i < N_MEANS; i++) {
		vdb_point(means[i][0],means[i][1],-1);
	}
	#endif
		
	double begin = current_time();
	
	for(int n = 0; n < MAX_ITERATIONS; n++) {
		bzero(new_means,sizeof(double) * N_MEANS * N_DIMS);
		bzero(count,sizeof(int) * N_MEANS * N_DIMS);
		for(int i = 0; i < N_POINTS; i++) {
			double min_dist_2 = DBL_MAX;
			int index = 0;
			for(int m = 0; m < N_MEANS; m++) {
				double dist_2 = 0.0;
				for(int d = 0; d < N_DIMS; d++) {
					double diff = points[i][d] - means[m][d];
					dist_2 += diff * diff;
				}
				if(dist_2 < min_dist_2) {
					min_dist_2 = dist_2;
					index = m;
				}
			}
			count[index]++;
			for(int d = 0; d < N_DIMS; d++)
				new_means[index][d] += (points[i][d] - new_means[index][d])/ count[index];
		}
		std::swap(means,new_means);
		#ifdef VDB_DEBUG
		for(int i = 0; i < N_MEANS; i++) {
			vdb_point(means[i][0],means[i][1],n);
		}
		#endif
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
