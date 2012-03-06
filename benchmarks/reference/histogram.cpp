

#include<stdio.h>
#include<stdlib.h>
#include<algorithm>
#include<math.h>

#include "timing.h"

int main() {
	int N = 10000000;
	double * data = new double[N];
	for(int i = 0; i < N; i++)
		data[i] = rand() % 100;
	
	double begin = current_time();

	int * result = new int[100];
	
	std::fill(result,result+100,0);
	
	for(int i = 0; i < N; i++) {
		result[(int)data[i]]++;
	}

	//for(int i = 0; i < 100; i++)
	//	printf("%d\n",result[i]);
	printf("Elapsed: %f\n", current_time()-begin);
	printf("%d\n", result[0]);
	return 0;
}
