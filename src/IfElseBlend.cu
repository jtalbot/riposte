
#include <stdio.h>
#include <cuda_runtime.h>

#define VECTOR_LENGTH (1<<24)
#define TILE_WIDTH 128		
#define NUM_BLOCKS 180

__global__ void IfElse(const int *control,const long *source1,const long *source2,long *dest, const long vectorLength) 
{	
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int stride = gridDim.x * blockDim.x;
	long idx = bid*blockDim.x + tid;		
	int elementsPerThread = ceil((float)vectorLength/stride);		

	for (int i = 0; i<elementsPerThread; ++i)
		if (idx < vectorLength) {
			if (control[idx])
				dest[idx] = source1[idx];
			else
				dest[idx] = source2[idx];
			idx += stride;
		}
}

// Helper function to handle CUDA errors
void ErrorHandle(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("Error with CUDA call (error code : %s)\n", cudaGetErrorString(err));
		exit(-1);
	}
}

int main(int argc, char **argv)
{
	
	long vectorLength = VECTOR_LENGTH;
	int numBlocks = NUM_BLOCKS;
	int numThreads = TILE_WIDTH;
	
	printf("Number of threads launched is %d\n", numBlocks*numThreads);
	printf("Elements per thread is %li\n", vectorLength/(numBlocks*numThreads));
	
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	
	// Allocate memory for host side vectors
	long sizeLogic = vectorLength*sizeof(int);
	long sizeData = vectorLength*sizeof(long);
	int * control = (int *)malloc(sizeLogic);
	long * source1 = (long *)malloc(sizeData);
	long * source2 = (long *)malloc(sizeData);
	long * dest = (long *)malloc(sizeData);
	if (control == NULL || source1 == NULL || source2 == NULL || dest == NULL) {
		printf("Unable to allocate host memory\n");
		return(-1);
	}
	
	for (int i = 0; i<vectorLength; ++i) {
		control[i] = i%2;
		source1[i] = 1;
		source2[i] = 2;
	}

	// GPU Vectors
	int * control_d;
	ErrorHandle(cudaMalloc((void **)&control_d, sizeLogic));
	ErrorHandle(cudaMemcpy(control_d, control, sizeLogic, cudaMemcpyHostToDevice));
	
	long * source1_d;
	ErrorHandle(cudaMalloc((void **)&source1_d, sizeData));
	ErrorHandle(cudaMemcpy(source1_d, source1, sizeData, cudaMemcpyHostToDevice));
	
	long * source2_d;
	ErrorHandle(cudaMalloc((void **)&source2_d, sizeData));
	ErrorHandle(cudaMemcpy(source2_d, source2, sizeData, cudaMemcpyHostToDevice));

	long * dest_d;
	ErrorHandle(cudaMalloc((void **)&dest_d, sizeData));

	
	
	
	
	
		
	// Kernel Invocation
	IfElse<<<dimGrid, dimBlock>>>(control_d, source1_d, source2_d, dest_d, vectorLength);
	ErrorHandle(cudaGetLastError());
	
	// Copy Results Back
	ErrorHandle(cudaMemcpy(dest, dest_d, sizeData, cudaMemcpyDeviceToHost));

	
	
	
	
	
	
	// Print Control
	printf("Control Vector\n[1]   ");
	for (int i = 0; i < 100; ++i)
		printf("%d  ",control[i]);
	printf("   . . .     ");
	for (int i = vectorLength-100; i < vectorLength; ++i)
		printf("%d  ",control[i]);
	printf("[%li]\n\n",vectorLength);
	
	// Print Results
	printf("Output Vector\n[1]   ");
	for (int i = 0; i < 100; ++i)
		printf("%li  ",dest[i]);
	printf("   . . .     ");
	for (int i = vectorLength-100; i < vectorLength; ++i)
		printf("%li  ",dest[i]);
	printf("[%li]\n",vectorLength);
	
	// Clean up
	ErrorHandle(cudaFree(control_d));
	ErrorHandle(cudaFree(source1_d));
	ErrorHandle(cudaFree(source2_d));
	ErrorHandle(cudaFree(dest_d));
	
	free(control);
	free(source1);
	free(source2);
	free(dest);
	
	return 0;
}