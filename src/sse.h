// Specialized SSE ops

#include <xmmintrin.h>

// Unary ops...

template<int64_t N>
struct Map1< NegOp<TDouble>, N > {
	static void eval(State& state, double const* a, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d xb = _mm_setzero_pd();
		__m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j++) {
                        xr[j] = _mm_sub_pd(xb, xa[j]);
                }
	}
};

static union ieee754_QNAN
{
        uint64_t i;
        double f;
        ieee754_QNAN() : i(0x7FFFFFFFFFFFFFFF) {}
} AbsMask;

static const __m128d absMask = _mm_load1_pd( &AbsMask.f);

template<int64_t N>
struct Map1< AbsOp<TDouble>, N > {
	static void eval(State& state, double const* a, double* r) {
                __m128d const* xa = (__m128d const*)a;
		__m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_and_pd(absMask, xa[j]);
                        xr[j+1] = _mm_and_pd(absMask, xa[j+1]);
                }
	}
};

template<int64_t N>
struct Map1< SqrtOp<TDouble>, N > {
	static void eval(State& state, double const* a, double* r) {
                __m128d const* xa = (__m128d const*)a;
		__m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_sqrt_pd(xa[j]);
                        xr[j+1] = _mm_sqrt_pd(xa[j+1]);
                }
	}
};



// Binary ops...

template<int64_t N>
struct Map2VV< AddOp<TDouble>, N > {
        static void eval(State& state, double const* a, double const* b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_add_pd(xa[j], xb[j]);
                        xr[j+1] = _mm_add_pd(xa[j+1], xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2SV< AddOp<TDouble>, N > {
        static void eval(State& state, double const a, double const* b, double* r) {
                __m128d xa = _mm_set1_pd(a);
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_add_pd(xa, xb[j]);
                        xr[j+1] = _mm_add_pd(xa, xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2VS< AddOp<TDouble>, N > {
        static void eval(State& state, double const* a, double const b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d xb = _mm_set1_pd(b);
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_add_pd(xa[j], xb);
                        xr[j+1] = _mm_add_pd(xa[j+1], xb);
                }
        }
};



template<int64_t N>
struct Map2VV< SubOp<TDouble>, N > {
        static void eval(State& state, double const* a, double const* b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_sub_pd(xa[j], xb[j]);
                        xr[j+1] = _mm_sub_pd(xa[j+1], xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2SV< SubOp<TDouble>, N > {
        static void eval(State& state, double const a, double const* b, double* r) {
                __m128d xa = _mm_set1_pd(a);
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_sub_pd(xa, xb[j]);
                        xr[j+1] = _mm_sub_pd(xa, xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2VS< SubOp<TDouble>, N > {
        static void eval(State& state, double const* a, double const b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d xb = _mm_set1_pd(b);
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_sub_pd(xa[j], xb);
                        xr[j+1] = _mm_sub_pd(xa[j+1], xb);
                }
        }
};



template<int64_t N>
struct Map2VV< MulOp<TDouble>, N > {
        static void eval(State& state, double const* a, double const* b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_mul_pd(xa[j], xb[j]);
                        xr[j+1] = _mm_mul_pd(xa[j+1], xb[j+1]);
 		}               
        }
};

template<int64_t N>
struct Map2SV< MulOp<TDouble>, N > {
        static void eval(State& state, double const a, double const* b, double* r) {
                __m128d xa = _mm_set1_pd(a);
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_mul_pd(xa, xb[j]);
                        xr[j+1] = _mm_mul_pd(xa, xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2VS< MulOp<TDouble>, N > {
        static void eval(State& state, double const* a, double const b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d xb = _mm_set1_pd(b);
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j] = _mm_mul_pd(xa[j], xb);
                        xr[j+1] = _mm_mul_pd(xa[j+1], xb);
                }
        }
};



template<int64_t N>
struct Map2VV< DivOp<TDouble>, N > {
        static void eval(State& state, double const* a, double const* b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j++) {
                        xr[j] = _mm_div_pd(xa[j], xb[j]);
                }
        }
};

template<int64_t N>
struct Map2SV< DivOp<TDouble>, N > {
        static void eval(State& state, double const a, double const* b, double* r) {
                __m128d xa = _mm_set1_pd(a);
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j++) {
                        xr[j] = _mm_div_pd(xa, xb[j]);
                }
        }
};

template<int64_t N>
struct Map2VS< DivOp<TDouble>, N > {
        static void eval(State& state, double const* a, double const b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d xb = _mm_set1_pd(b);
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j++) {
                        xr[j] = _mm_div_pd(xa[j], xb);
                }
        }
};

