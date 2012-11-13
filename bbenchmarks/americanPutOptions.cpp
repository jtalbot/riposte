
#include <math.h>
#include <stdint.h>

#include <vector>

double* americanPut(double T, double* S, double K, double r, double sigma, double q, int64_t n) {
    double deltaT = T/n;
    double up = exp(sigma*sqrt(deltaT));
    double p0 = (up * exp(-r*deltaT) - exp(-q*deltaT)) * up / (up*up-1);
    double p1 = exp(-r*deltaT) - p0;

    std::vector<double*> p;
    for(int i = 0; i < n; i++) {
        p.push_back(
    }
}
