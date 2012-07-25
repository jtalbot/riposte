// Simple CUDA Program
// Sum Reduction Operation

#include <stdio.h>
#include <cuda_runtime.h>

#define VECTOR_LENGTH (1<<28)
#define TILE_WIDTH 128		

// Partially sums an input vector into a smaller vector with number of elements = number of launched blocks
__global__ void Sum(const int *vector, long *resultVector, const long vectorLength) 
{	
	// indexing variables
	int stride = gridDim.x * blockDim.x;				// number of threads total
	long idx = blockIdx.x*blockDim.x + threadIdx.x;		// global thread index
	int tx = threadIdx.x;								// local thread index
	int bx = blockDim.x;								// block width (threads)
	int elementsPerThread = vectorLength/(stride);		// number of elements each thread must calculate
	
	// TODO Fix this line (don't use macro value)
	// Has one accumulator value for each thread
	__shared__ long blockSum[TILE_WIDTH];
	// local accumulator
	long sum = 0;
	
	// sum the values allocated for this thread
	// (from global memory)
	for (int i = 0; i<elementsPerThread; ++i) {
		sum += vector[idx + stride*i];
		printf("%d\n",idx+stride*i);
	}
	
	// copy local result to shared memory
	blockSum[tx] = sum;
	__syncthreads();
	
	// accumulate shared memory sums in thread 0
	// copy to output global vector
	if (tx == 0) {
		for (int i = 1; i < bx; ++i)
			sum += blockSum[i];
		// write to output global memory
		resultVector[blockIdx.x] = sum;
	}

}

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

int main(int argc, char **argv)
{
	
	long vectorLength = VECTOR_LENGTH;
	int numBlocks = 180;
	
	printf("Number of threads launched is %d\n", numBlocks*TILE_WIDTH);
	printf("Elements per thread is %li\n", vectorLength/(numBlocks*TILE_WIDTH));
	
	dim3 dimBlock(TILE_WIDTH, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	
	// Allocate host memory for input vector and partial vector returned from GPU
	int size = vectorLength * sizeof(int);
	int * vector = (int *)malloc(size);
	int partSize = numBlocks*sizeof(long);
	long * partVector = (long *)malloc(partSize);
	if (vector == NULL || partVector == NULL) {
		printf("Unable to allocate host memory\n");
		return(-1);
	}
	
	// Fill the input vector with random 0's and 1's
	time_t seed;
	time(&seed);
	srand((unsigned int) seed);
	for (int i = 0; i<vectorLength; ++i) {
		vector[i] = rand()%2;
	}
	
	// Allocate GPU memory for the input vector and output partial vector
	int * vector_d;
	ErrorHandle(cudaMalloc((void **)&vector_d, size));
	ErrorHandle(cudaMemcpy(vector_d, vector, size, cudaMemcpyHostToDevice));
	
	long * result_d;
	//int size = numBlocks * sizeof(long);
	ErrorHandle(cudaMalloc((void **)&result_d, partSize));
	
	long sum1 = 0;
	long sum2 = 0;
	
	// run operation 100 times and average for timing
	float elapsedTime = 0;	
	
	cudaEvent_t start, stop;
	ErrorHandle(cudaEventCreate(&start));
	ErrorHandle(cudaEventCreate(&stop));
	for (int i = 0; i < 100; ++i) {
		if (i > 9) {
			ErrorHandle(cudaEventRecord(start, 0));
		}
		
		// Kernel Invocation
		Sum<<<dimGrid, dimBlock>>>(vector_d, result_d, vectorLength);
		ErrorHandle(cudaGetLastError());
		
		// pass back to CPU for remainder of reduction
		ErrorHandle(cudaMemcpy(partVector, result_d, partSize, cudaMemcpyDeviceToHost));
		sum1 = sumVector(partVector, numBlocks);
		
		if (i > 9) {
			ErrorHandle(cudaEventRecord(stop, 0));
			ErrorHandle(cudaEventSynchronize(stop));		
			float elapsedT;
			ErrorHandle(cudaEventElapsedTime(&elapsedT, start, stop));	
			elapsedTime += elapsedT/90000;
		}
	}
	ErrorHandle(cudaEventDestroy(start));			
	ErrorHandle(cudaEventDestroy(stop));
	
	printf("Time Elapsed for Kernel Operation is %lf seconds\n", elapsedTime);
	double thruPut = (double)(sizeof(int)*vectorLength) / (1024 * 1024 * 1024 * elapsedTime);
	printf("Data Rate is %lf GB/s\n", thruPut);
	printf("GPU Result : %li\n", sum1);
	
	// CPU comparison
	for (long i = 0; i<vectorLength; i++) 
		sum2 += vector[i];
	printf("CPU Result : %li\n", sum2);
	
	// Clean up
	ErrorHandle(cudaFree(vector_d));
	ErrorHandle(cudaFree(result_d));
	
	free(vector);
	
	return 0;
}
