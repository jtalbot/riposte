
#include <smmintrin.h>
#include <math.h>
#include "value.h"
#include "interpreter.h"

struct SSEValue {
        union {
                __m128d D;
                __m128i I;
                int64_t i[2];
                double  d[2];
        };
};

// ARITH_UNARY

extern "C"
__m128i neg_i(__m128i a) {
	SSEValue v;
	v.I = a;
	for(int i = 0; i < 2; i++) {
		v.i[i] = -v.i[i];
	}
	return v.I;
}

extern "C"
__m128i abs_i(__m128i a) {
        SSEValue v; 
        v.I = a;
	for(int i = 0; i < 2; i++) 
		v.i[i] = v.i[i] < 0 ? -v.i[i] : v.i[i];
        return v.I;
}

extern "C"
__m128d sign_d(__m128d input) {
	SSEValue v;
	v.D = input;
	for(int i = 0; i < 2; i++) 
		v.d[i] = v.d[i] > 0 ? 1 : 
			(v.d[i] < 0 ? -1 :
			(v.d[i] == 0 ? 0 : Double::NAelement));
	return v.D; 
}

extern "C" ALWAYS_INLINE 
__m128d sqrt_d(__m128d input) {
	return _mm_sqrt_pd(input);
}

extern "C" ALWAYS_INLINE 
__m128d floor_d(__m128d input) {
	return _mm_round_pd(input, 0x1);
}

extern "C" ALWAYS_INLINE 
__m128d ceiling_d(__m128d input) {
	return _mm_round_pd(input, 0x2);
}

extern "C" ALWAYS_INLINE 
__m128d trunc_d(__m128d input) {
	return _mm_round_pd(input, 0x3);
}

extern "C" ALWAYS_INLINE 
__m128d exp_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = exp(v.d[i]);
       return v.D;
}

extern "C" ALWAYS_INLINE 
__m128d log_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = log(v.d[i]);
       return v.D;
}

extern "C" ALWAYS_INLINE 
__m128d cos_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = cos(v.d[i]);
       return v.D;
}

extern "C" ALWAYS_INLINE 
__m128d sin_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = sin(v.d[i]);
       return v.D;
}

extern "C" ALWAYS_INLINE 
__m128d tan_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = tan(v.d[i]);
       return v.D;
}

extern "C" ALWAYS_INLINE 
__m128d acos_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = acos(v.d[i]);
       return v.D;
}

extern "C" ALWAYS_INLINE 
__m128d asin_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = asin(v.d[i]);
       return v.D;
}

extern "C" ALWAYS_INLINE 
__m128d atan_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = atan(v.d[i]);
       return v.D;
}


// ARITH_BINARY

/*extern "C"
__m128i add_i(__m128i a, __m128i b) {
        SSEValue v, w; 
        v.I = a;
        w.I = b;
	for(int i = 0; i < 2; i++) {
		if((w.i[i] > 0) && (v.i[i] > std::numeric_limits<int64_t>::max() - w.i[i]))
			v.i[i] = Integer::NAelement;
		else if((w.i[i] < 0) && (v.i[i] <= std::numeric_limits<int64_t>::min() - w.i[i])) 
			v.i[i] = Integer::NAelement;
		else
			v.i[i] = INTEGER2_NA(v.i[i], w.i[i], v.i[i] + w.i[i]);
        }
	return v.I;
}*/

extern "C"
__m128i mod_i(__m128i a, __m128i b) {
        SSEValue v, w; 
        v.I = a;
        w.I = b;
	for(int i = 0; i < 2; i++) 
		v.i[i] = v.i[i] % w.i[i];
        return v.I;
}

extern "C"
__m128i idiv_i(__m128i a, __m128i b) {
        SSEValue v, w; 
        v.I = a;
        w.I = b;
	for(int i = 0; i < 2; i++) 
		v.i[i] = v.i[i] / w.i[i];
        return v.I;
}

extern "C"
__m128i pmin_i(__m128i a, __m128i b) {
        SSEValue v, w; 
        v.I = a;
        w.I = b;
	for(int i = 0; i < 2; i++) 
		v.i[i] = v.i[i] < w.i[i] ? v.i[i] : w.i[i];
        return v.I;
}

extern "C"
__m128i pmax_i(__m128i a, __m128i b) {
        SSEValue v, w; 
        v.I = a;
        w.I = b;
	for(int i = 0; i < 2; i++) 
		v.i[i] = v.i[i] > w.i[i] ? v.i[i] : w.i[i];
        return v.I;
}

extern "C"
int64_t pmin_i1(int64_t a, int64_t b) {
	return std::min(a,b);
}

extern "C"
int64_t pmax_i1(int64_t a, int64_t b) {
	return std::max(a,b);
}

extern "C"
__m128d pow_d(__m128d a, __m128d b) {
        SSEValue v, w; 
        v.D = a;
        w.D = b;
	for(int i = 0; i < 2; i++) 
		v.d[i] = pow(v.d[i],w.d[i]);
        return v.D;
}

extern "C"
__m128d atan2_d(__m128d a, __m128d b) {
        SSEValue v, w; 
        v.D = a;
        w.D = b;
	for(int i = 0; i < 2; i++) 
		v.d[i] = atan2(v.d[i],w.d[i]);
        return v.D;
}

extern "C"
__m128d hypot_d(__m128d a, __m128d b) {
        SSEValue v, w; 
        v.D = a;
        w.D = b;
	for(int i = 0; i < 2; i++) 
		v.d[i] = hypot(v.d[i],w.d[i]);
        return v.D;
}

extern "C"
__m128d pmin_d(__m128d a, __m128d b) {
	return _mm_min_pd(a, b);
}

extern "C"
__m128d pmax_d(__m128d a, __m128d b) {
	return _mm_max_pd(a, b);
}

extern "C"
double pmin_d1(double a, double b) {
	return std::min(a,b);
}

extern "C"
double pmax_d1(double a, double b) {
	return std::max(a,b);
}

extern "C"
__m128d random_d(int64_t thread) {
        Thread::RandomSeed& r = Thread::seed[thread];
        
        // advance three times to avoid taking powers of 2
        r.v[0] = r.v[0] * r.m[0] + r.a[0];
        r.v[1] = r.v[1] * r.m[1] + r.a[1];
        r.v[0] = r.v[0] * r.m[0] + r.a[0];
        r.v[1] = r.v[1] * r.m[1] + r.a[1];
        r.v[0] = r.v[0] * r.m[0] + r.a[0];
        r.v[1] = r.v[1] * r.m[1] + r.a[1];

        SSEValue o;

        o.d[0] = (double)r.v[0] / ((double)std::numeric_limits<uint64_t>::max() + 1);
        o.d[1] = (double)r.v[1] / ((double)std::numeric_limits<uint64_t>::max() + 1);
        return o.D;
}

extern "C"
__m128i isna_d(__m128d a) {
        SSEValue v; 
        v.D = a;
	for(int i = 0; i < 2; i++) 
		v.i[i] = Double::isNA(v.d[i]) ? -1 : 0;
        return v.I;
}

extern "C"
__m128i isnan_d(__m128d a) {
        SSEValue v; 
        v.D = a;
	for(int i = 0; i < 2; i++) 
		v.i[i] = Double::isNaN(v.d[i]) ? -1 : 0;
        return v.I;
}

extern "C"
__m128i isfinite_d(__m128d a) {
        SSEValue v; 
        v.D = a;
	for(int i = 0; i < 2; i++) 
		v.i[i] = Double::isFinite(v.d[i]) ? -1 : 0;
        return v.I;
}

extern "C"
__m128i isinfinite_d(__m128d a) {
        SSEValue v; 
        v.D = a;
	for(int i = 0; i < 2; i++) 
		v.i[i] = Double::isInfinite(v.d[i]) ? -1 : 0;
        return v.I;
}

