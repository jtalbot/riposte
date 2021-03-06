
/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/
static inline int mandel(double c_re, double c_im, int count) {
    double z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.)
            break;

        double new_re = z_re*z_re - z_im*z_im;
        double new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

export void mandelbrot_loop(uniform double x0, uniform double y0, 
                            uniform double x1, uniform double y1,
                            uniform int width, uniform int height, 
                            uniform int maxIterations,
                            uniform int output[])
{
    double dx = (x1 - x0) / width;
    double dy = (y1 - y0) / height;

    for (uniform int j = 0; j < height; j++) {
        // Note that we'll be doing programCount computations in parallel,
        // so increment i by that much.  This assumes that width evenly
        // divides programCount.
        foreach (i = 0 ... width) {
            // Figure out the position on the complex plane to compute the
            // number of iterations at.  Note that the x values are
            // different across different program instances, since its
            // initializer incorporates the value of the programIndex
            // variable.
            double x = x0 + i * dx;
            double y = y0 + j * dy;

            int index = j * width + i;
            output[index] = mandel(x, y, maxIterations);
        }
    }
}

#define N_DIMS 2
#define N_MEANS 5
#define N_POINTS 1000000
#define DBL_MAX 10000000000000.0
export void kmeans_loop(uniform double points[], uniform int32 count[], uniform double means[], uniform double new_means[]) {
	for(uniform int ii = 0; ii < N_POINTS; ii += programCount) {
		int i = ii + programIndex;
		double min_dist_2 = DBL_MAX;
		int index = 0;
		for(uniform int m = 0; m < N_MEANS; m++) {
			double dist_2 = 0.0;
			for(uniform int d = 0; d < N_DIMS; d++) {
				uniform double mm = means[N_DIMS * m + d];
				double diff = points[N_POINTS * d + i] - mm;
				dist_2 += diff * diff;
			}
			if(dist_2 < min_dist_2) {
				min_dist_2 = dist_2;
				index = m;
			}
		}
		count[programCount * N_MEANS +index]++;
		for(uniform int d = 0; d < N_DIMS; d++)
			new_means[programCount * N_MEANS * N_DIMS + index * N_DIMS + d] += (points[d * N_POINTS + i] - new_means[programCount * N_MEANS * N_DIMS + index * N_DIMS + d]) / count[programCount * N_MEANS +index];
	}
}


export void smv_loop(uniform int M, uniform int64 col_idx[], uniform int64 row_idx[], uniform double values[],uniform double v[],uniform double result[]) {
	for(uniform int ii = 0; ii < M; ii += programCount) {
		int i = ii + programIndex;
		int c = col_idx[i];
		int r = row_idx[i];
		result[r] += values[i] * v[c];
	}
}

export uniform double raysphere_loop(uniform int n,uniform double start, uniform double xd, uniform double yd, uniform double zd, uniform double xo, uniform double xc[], uniform double yo, uniform double yc[], uniform double zo, uniform double zc[]) {
	double r = start;
	for(uniform int ii = 0; ii < n; ii += programCount) {
		int i = ii + programIndex;
		double b = 2*(xd*(xo-xc[i])+yd*(yo-yc[i])+zd*(zo-zc[i]));
		double c = (xo-xc[i])*(xo-xc[i])+(yo-yc[i])*(yo-yc[i])+(zo-zc[i])*(zo-zc[i])-1;
		
		double disc = b*b-4*c;
		if(disc > 0) {
			double t0 = (-b - sqrt(disc))/2;
			double t1 = (-b + sqrt(disc))/2;
			r = min(r,min(t0,t1));
		}
	}
	return reduce_min(r);
}