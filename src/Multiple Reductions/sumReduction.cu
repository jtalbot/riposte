// Simple CUDA Program
// Sum Reduction Operation

#include <stdio.h>
#include <cuda_runtime.h>
#include "input.h"
#include "sumReduction.h"
#include "kernels.h"

int main(int argc, char **argv)
{
	// Set up kernel for launch
	long vectorLength = VECTOR_LENGTH;
	int numBlocks = NUM_BLOCKS;
	int numThreads = TILE_WIDTH;
	int numVectors = NUM_VECT;
	
	printf("Number of threads launched is %d\n", numBlocks*numThreads);
	printf("Elements of each vector per thread is %li\n", vectorLength/(numBlocks*numThreads));
	
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	
	// Allocate host memory for input vector and partial vector returned from GPU
	long size = vectorLength * numVectors * sizeof(long);
	long * vector = (long *)malloc(size);
	int partSize = numBlocks * numVectors * sizeof(long);
	long * partVector = (long *)malloc(partSize);
	long * tempVectorGPU;
	long * tempVectorCPU;
	if (vector == NULL || partVector == NULL) {
		printf("Unable to allocate host memory\n");
		return(-1);
	}
	
	// Fill the input vectors with random 1's
	for (int i = 0; i<(vectorLength * numVectors); ++i)
		vector[i] = rand()%2;
	
	// Allocate GPU memory for the input vectors and output partial vectors
	long * vector_d;
	ErrorHandle(cudaMalloc((void **)&vector_d, size));
	ErrorHandle(cudaMemcpy(vector_d, vector, size, cudaMemcpyHostToDevice));
	
	long * result_d;
	ErrorHandle(cudaMalloc((void **)&result_d, partSize));
		
	long sumGPU[numVectors];
	long sumCPU[numVectors];
			
	// Launch the kernel (either shared memory or global memory version, based on number of input vectors)
	if ((numVectors * numBlocks * sizeof(long)) < MAX_SHARED_MEM) {
		printf("Using Shared Memory\n");
		size_t sharedMem = numVectors * numThreads * sizeof(long);
		SumShared<<<dimGrid, dimBlock, sharedMem>>>(vector_d, result_d, vectorLength, numVectors);
	} else {
		printf("Using Global Memory\n");
		long * interValues_d;
		ErrorHandle(cudaMalloc((void **)&interValues_d, numVectors*numBlocks*numThreads*sizeof(long)));
		SumGlobal<<<dimGrid, dimBlock>>>(vector_d, result_d, interValues_d, vectorLength, numVectors);
		ErrorHandle(cudaFree(interValues_d));
	}

	ErrorHandle(cudaGetLastError());

	// pass back to CPU for remainder of reduction
	ErrorHandle(cudaMemcpy(partVector, result_d, partSize, cudaMemcpyDeviceToHost));
	for (int i = 0; i<numVectors; ++i) {
		tempVectorGPU = (partVector + i*numBlocks);
		sumGPU[i] = sumVector(tempVectorGPU, numBlocks);
	}
	
	// CPU comparison
	for (int i = 0; i<numVectors; ++i) {
		tempVectorCPU = (vector + i*vectorLength);
		sumCPU[i] = sumVector(tempVectorCPU, vectorLength);
	}
	
	// Verify results
	for (int i = 0; i<numVectors; ++i) {
		printf("Verifying results for vector %d . . . ", i);
		if (!(sumCPU[i]-sumGPU[i]))
			printf("Verified\n");
		else 
			printf("Error Detected\n");
	}
	
	// Clean up memory
	ErrorHandle(cudaFree(vector_d));
	ErrorHandle(cudaFree(result_d));
	
	free(vector);
	free(partVector);
	
	return 0;
}

