#include<stdio.h>
#include<math.h>

double R[4][4] = {  12,    6, -4, 1,
                   -51,  167, 24, 2,
                     4,  -68,-41, 4,
                     5,  -70,-40, 3};
double Q[4][4] = { 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1};

static const int N_ROWS = 4;
static const int N_COLS = 4;

double sign(double a) {
	if(a > 0)
		return 1;
	else if (a < 0)
		return -1;
	else
		return 0;
}

int main() {
	double v[N_ROWS];
	for(int c = 0; c < N_COLS-1; c++) {
		double lcl = 0.0;
		
		/*
		for(int r = 0; r < c; r++) {
			v[r] = 1.0/0.0;
		}*/
		
		double b = 0.0;
		lcl += R[c][c] * R[c][c];
		
		for(int r = c + 1; r < N_ROWS; r++) {
			double tmp = R[c][r] * R[c][r];
			v[r] = R[c][r];
			lcl += tmp;
			b += tmp;
		}
		lcl = sqrt(lcl);
		//printf("lcl = %f\n",lcl);
		double a = R[c][c] + sign(R[c][c]) * lcl;
		v[c] = a;
		b += a * a;
		//printf("b = %f\n", b);
		double two_on_b = 2.0 / b;
		
		double v_t_R[N_COLS];
		double Q_v[N_ROWS];
		
		for(int cc = 0; cc < N_COLS; cc++) {
			double tmp = 0.0;
			for(int rr = c; rr < N_ROWS; rr++) {
				tmp += R[cc][rr] * v[rr];
			}
			v_t_R[cc] = tmp;
			//printf("v_t_R[%d] = %f\n",cc,v_t_R[cc]);
		}
		
		for(int rr = 0; rr < N_ROWS; rr++) {
			double tmp = 0.0;
			for(int cc = c; cc < N_COLS; cc++) {
				tmp += Q[rr][cc] * v[cc];
			}
			Q_v[rr] = tmp;
			//printf("Q_v[%d] = %f\n",rr,Q_v[rr]);
		}
		
		for(int cc = 0; cc < N_COLS; cc++) {
			for(int rr = c; rr < N_ROWS; rr++) {
				R[cc][rr] -= two_on_b * v[rr] * v_t_R[cc];
			}
		}
		for(int cc = c; cc < N_COLS; cc++) {
			for(int rr = 0; rr < N_ROWS; rr++) {
				Q[rr][cc] -= two_on_b * Q_v[rr] * v[cc];
			}
		}
	}
	
	for(int rr = 0; rr < N_ROWS; rr++) {
		for(int cc = 0; cc < N_COLS; cc++) {
			printf("%f ",Q[rr][cc]);
		}
		printf("\n");
	}
	for(int rr = 0; rr < N_ROWS; rr++) {
		for(int cc = 0; cc < N_COLS; cc++) {
			printf("%f ",R[cc][rr]);
		}
		printf("\n");
	}
	return 0;
}