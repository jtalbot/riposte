/*
 *  kernels.h
 *  
 *
 *  Created by dylanz on 8/1/12.
 *  Copyright 2012 Stanford University. All rights reserved.
 *
 */

#ifndef KERNELS_H
#define KERNELS_H
#include "input.h"

//===KERNEL FUNCTIONS============================================================================================
//===============================================================================================================

// Partially sums input vectors (in parallel) into smaller, 180 element vectors.
// This version of the kernel holds the intermediate values in shared memory
__global__ void SumShared(const long *vector, long *resultVector, const long vectorLength, const int numVectors) 
{	
	// indexing variables
	int stride = gridDim.x * blockDim.x;				
	long idx = blockIdx.x*blockDim.x + threadIdx.x;		
	int tx = threadIdx.x;								
	int bx = blockDim.x;								
	int elementsPerThread = ceil((float)vectorLength/stride);		
	
	// local accumulator
	// TODO fix this value (no macro)
	long sum[NUM_VECT] = { 0 };
	
	// sum the values allocated for this thread
	// (from global memory)
	for (int i = 0; i<elementsPerThread; ++i)
		if (idx < vectorLength) {
			for (int j = 0; j < numVectors; ++j)
				sum[j] += vector[idx + j*vectorLength];
			idx += stride;
		}
	
	// Has one accumulator value for each thread
	extern __shared__ long blockSum[];
	
	// copy local results to shared memory
	for (int j = 0; j < numVectors; ++j) {
		blockSum[j*bx + tx] = sum[j];
		sum[j] = 0;
	}
	__syncthreads();
	
	// Copy results back to global memory
	// Each vector summation is handled by one thread
	int numPasses = ceil((float)numVectors/bx);
	for (int j = 0; j < numPasses; ++j)
		if ((tx+j*bx) < numVectors) {
			for (int i = 0; i < bx; ++i)
				sum[tx+j*bx] += blockSum[(tx+j)*bx+i];						
			resultVector[blockIdx.x + (tx+j)*gridDim.x] = sum[tx+j*bx];
		}

	
}

// Partially sums input vectors (in parallel) into smaller, 180 element vectors.
// This version of the kernel holds the intermediate values in global memory
__global__ void SumGlobal(const long *vector, long *resultVector, long *interValues, const long vectorLength, const int numVectors) 
{	
	// indexing variables
	int stride = gridDim.x * blockDim.x;				
	long idx = blockIdx.x*blockDim.x + threadIdx.x;		
	int tx = threadIdx.x;								
	int bx = blockDim.x;								
	int elementsPerThread = ceil((float)vectorLength/stride);		
	
	// local accumulator
	// TODO fix this value (no macro)
	long sum[NUM_VECT] = { 0 };
	
	// sum the values allocated for this thread
	// (from global memory)
	for (int i = 0; i<elementsPerThread; ++i)
		if (idx < vectorLength) {
			for (int j = 0; j < numVectors; ++j)
				sum[j] += vector[idx + j*vectorLength];
			idx += stride;
		}
	
	
	// copy local results to global memory
	for (int j = 0; j < numVectors; ++j) {
		interValues[tx+blockIdx.x*bx+j*stride] = sum[j];
		sum[j] = 0;
	}
	__syncthreads();
	
	// Sum results then store in final global memory location
	// Each vector has its block sums computed by a single thread
	int numPasses = ceil((float)numVectors/bx);
	for (int j = 0; j < numPasses; ++j)
		if ((tx+j*bx) < numVectors) {
			for (int i = 0; i < bx; ++i)
				sum[tx+j*bx] += interValues[(tx+j*bx)*stride+blockIdx.x*bx+i];						
			resultVector[blockIdx.x+tx*gridDim.x+j*stride] = sum[tx+j*bx];
		}	
}


#endif