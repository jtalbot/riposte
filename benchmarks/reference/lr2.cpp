

#define N_DIMS 2
#define N_ROWS 1000
#define N_ITERATIONS 1000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#define VIEW
#ifdef VIEW
#include <vdb.h>
#endif
double drand() {
	double r = arc4random() / (double)0xFFFFFFFF;
	return r;
}
int main() {
	double * w = new double[N_DIMS + 1];
	bzero(w, sizeof(double) * (N_DIMS + 1));
	double (*data)[N_DIMS] = (double(*)[N_DIMS]) malloc(sizeof(double) * N_DIMS * N_ROWS);
	double * response = new double[N_ROWS];
	
	for(int i = 0; i < N_ROWS; i++) {
		double p = drand();
		response[i] = 3;
		for(int d = 0; d < N_DIMS; d++) {
			data[i][d] = drand();
			response[i] += (d + 1)*data[i][d] + (drand() - .5) * .1; 
		}
		#ifdef VIEW
		vdb_color(1,1,1);
		vdb_point(data[i][0],data[i][1],response[i]);
		#endif
	}
	
	double * grad = new double[N_DIMS + 1];
	for(int round = 0; round < N_ITERATIONS; round++) {
		bzero(grad,sizeof(double) * (N_DIMS + 1));
		
		for(int r = 0; r < N_ROWS; r++) {
			double diff = w[0];
			for(int d = 0; d < N_DIMS; d++) {
				diff += w[d+1] * data[r][d];
			}
			diff -= response[r];
			grad[0] += (diff - grad[0]) / (r + 1);
			for(int d = 0; d < N_DIMS + 1; d++) {
				grad[d+1] += (diff*data[r][d] - grad[d+1]) / (r + 1);
			}
		}
		
		for(int d = 0; d < N_DIMS + 1; d++) {
			w[d] -= .07 * grad[d];
		}
	}
	
	#ifdef VIEW
	vdb_color(1,0,0);
	vdb_triangle(0,0,w[0],1,0,w[0]+ w[1],0,1,w[0]+w[2]);
	#endif
	
	return 0;
	
}

