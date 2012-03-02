
#include <stdio.h>

double v[] = { 1, 2 , 3 , 4};

int    row_idx[] = {0,1,3,4,5}; 
int    columns[] = {0,0,1,2,3};
double values[] = {2.5,1.5,1,9.5,1};


int main() {
	double result[4];
	
	for(int r = 0; r < 4; r++) {
		double a = 0;
		for(int i = row_idx[r]; i < row_idx[r+1]; i++) {
			int c = columns[i];
			a += values[i] * v[c]; 
		}
		result[r] = a;
	}
	printf("%f %f %f %f\n",result[0],result[1],result[2],result[3]);
}