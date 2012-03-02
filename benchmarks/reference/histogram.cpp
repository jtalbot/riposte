

#include<stdio.h>
#include<stdlib.h>
#include<algorithm>
#include<math.h>

int main() {
	int N = 100000;
	double * data = new double[N];
	for(int i = 0; i < N; i++)
		data[i] = arc4random() % 100 + 1;
	
	int * result = new int[100];
	
	std::fill(result,result+100,0);
	
	for(int i = 0; i < N; i++) {
		result[(int)floor(data[i])-1]++;
	}
	
	for(int i = 0; i < 100; i++)
		printf("%d\n",result[i]);
	return 0;
}