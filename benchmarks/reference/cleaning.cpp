#include <cmath>
#include <stdio.h>

bool is_outlier(double x) { return std::isnan(x) || x == 9999; }


int main() {
	const int N = 20000000;
	double * data = new double[N];
	
	for(int i = 0; i < N; i++)
		data[i] = i + 1;
		
	
	double mean = 0.0;
	double mean_2 = 0.0;
	
	int N_filtered = 0;
	for(int i = 0; i < N; i++) {
		if(!is_outlier(data[i])) {
			N_filtered++;
			mean += (data[i] - mean) / N_filtered;
			mean_2 += data[i] * data[i];
		}
	}
	double stddev = sqrt( (mean_2 / N_filtered  - mean * mean) * N_filtered / (N_filtered - 1));
	
	//printf("%f %f\n", mean,stddev);
	double r = 0.0;
	for(int i = 0; i < N; i++) {
		if(!is_outlier(data[i]))
			r += fabs((data[i] - mean) / stddev) > 1;
	}
	printf("%f\n",r);
}
