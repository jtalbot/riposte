/*
 *  sumReduction.h
 *  
 *
 *  Created by dylanz on 8/1/12.
 *  Copyright 2012 Stanford University. All rights reserved.
 *
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

//===HELPER FUNCTIONS============================================================================================
//===============================================================================================================
// Helper function to handle CUDA errors
void ErrorHandle(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("Error with CUDA call (error code : %s)\n", cudaGetErrorString(err));
		exit(-1);
	}
}

// Function that sums a vector of long ints on the CPU
long sumVector(long * vector, long vectorLength) {	
	long sum = 0;	
	for (long i = 0; i<vectorLength; ++i)
		sum += vector[i];	
	return sum;
}
#endif