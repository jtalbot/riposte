// Specialized SSE ops

#include <xmmintrin.h>

// Unary ops...
#ifdef SSE_OPS
template<int64_t N>
struct Map1< NegOp<TDouble>, N, true > {
	static void eval(State& state, double const* a, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d xb = _mm_setzero_pd();
		__m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_sub_pd(xb, xa[j+0]);
                        xr[j+1] = _mm_sub_pd(xb, xa[j+1]);
                }
	}
};

template<int64_t N>
struct Map1< AbsOp<TDouble>, N, true > {
	
	union ieee754_QNAN
	{
	        uint64_t i;
	        double f;
	        ieee754_QNAN() : i(0x7FFFFFFFFFFFFFFF) {}
	};
	static void eval(State& state, double const* a, double* r) {
                const ieee754_QNAN AbsMask;
		const __m128d absMask = _mm_load1_pd( &AbsMask.f);
		__m128d const* xa = (__m128d const*)a;
		__m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_and_pd(absMask, xa[j+0]);
                        xr[j+1] = _mm_and_pd(absMask, xa[j+1]);
                }
	}
};

template<int64_t N>
struct Map1< SqrtOp<TDouble>, N, true > {
	static void eval(State& state, double const* a, double* r) {
                __m128d const* xa = (__m128d const*)a;
		__m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_sqrt_pd(xa[j+0]);
                        xr[j+1] = _mm_sqrt_pd(xa[j+1]);
                }
	}
};

#ifdef USE_AMD_LIBM
#include <amdlibm.h>
template<int64_t N>
struct Map1< ExpOp<TDouble>, N, true > {
	static void eval(State& state, double const* a, double* r) {
        __m128d const* xa = (__m128d const*)a;
		__m128d* xr = (__m128d*)r;
		for(int j = 0; j < N/2; j+=2) {
			xr[j+0] = amd_vrd2_exp(xa[j+0]);
			xr[j+1] = amd_vrd2_exp(xa[j+1]);
		}
	}
};
template<int64_t N>
struct Map1< LogOp<TDouble>, N, true > {
	static void eval(State& state, double const* a, double* r) {
        __m128d const* xa = (__m128d const*)a;
		__m128d* xr = (__m128d*)r;
		for(int j = 0; j < N/2; j+=2) {
			xr[j+0] = amd_vrd2_log(xa[j+0]);
			xr[j+1] = amd_vrd2_log(xa[j+1]);
		}
	}
};
#endif


// Binary ops...

template<int64_t N>
struct Map2VV< AddOp<TDouble>, N, true > {
        static void eval(State& state, double const* a, double const* b, double* r) {
		__m128d const* xa = (__m128d const*)a;
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_add_pd(xa[j+0], xb[j+0]);
                        xr[j+1] = _mm_add_pd(xa[j+1], xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2SV< AddOp<TDouble>, N, true > {
        static void eval(State& state, double const a, double const* b, double* r) {
                const __m128d xa = _mm_set1_pd(a);
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_add_pd(xa, xb[j+0]);
                        xr[j+1] = _mm_add_pd(xa, xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2VS< AddOp<TDouble>, N, true > {
        static void eval(State& state, double const* a, double const b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                const __m128d xb = _mm_set1_pd(b);
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_add_pd(xa[j+0], xb);
                        xr[j+1] = _mm_add_pd(xa[j+1], xb);
                }
        }
};



template<int64_t N>
struct Map2VV< SubOp<TDouble>, N, true > {
        static void eval(State& state, double const* a, double const* b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_sub_pd(xa[j+0], xb[j+0]);
                        xr[j+1] = _mm_sub_pd(xa[j+1], xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2SV< SubOp<TDouble>, N, true > {
        static void eval(State& state, double const a, double const* b, double* r) {
                const __m128d xa = _mm_set1_pd(a);
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_sub_pd(xa, xb[j+0]);
                        xr[j+1] = _mm_sub_pd(xa, xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2VS< SubOp<TDouble>, N, true > {
        static void eval(State& state, double const* a, double const b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                const __m128d xb = _mm_set1_pd(b);
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_sub_pd(xa[j+0], xb);
                        xr[j+1] = _mm_sub_pd(xa[j+1], xb);
                }
        }
};



template<int64_t N>
struct Map2VV< MulOp<TDouble>, N, true > {
        static void eval(State& state, double const* a, double const* b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_mul_pd(xa[j+0], xb[j+0]);
                        xr[j+1] = _mm_mul_pd(xa[j+1], xb[j+1]);
 		}               
        }
};

template<int64_t N>
struct Map2SV< MulOp<TDouble>, N, true > {
        static void eval(State& state, double const a, double const* b, double* r) {
                const __m128d xa = _mm_set1_pd(a);
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_mul_pd(xa, xb[j+0]);
                        xr[j+1] = _mm_mul_pd(xa, xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2VS< MulOp<TDouble>, N, true > {
        static void eval(State& state, double const* a, double const b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                const __m128d xb = _mm_set1_pd(b);
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_mul_pd(xa[j+0], xb);
                        xr[j+1] = _mm_mul_pd(xa[j+1], xb);
                }
        }
};



template<int64_t N>
struct Map2VV< DivOp<TDouble>, N, true > {
        static void eval(State& state, double const* a, double const* b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_div_pd(xa[j+0], xb[j+0]);
                        xr[j+1] = _mm_div_pd(xa[j+1], xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2SV< DivOp<TDouble>, N, true > {
        static void eval(State& state, double const a, double const* b, double* r) {
                const __m128d xa = _mm_set1_pd(a);
                __m128d const* xb = (__m128d const*)b;
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_div_pd(xa, xb[j+0]);
                        xr[j+1] = _mm_div_pd(xa, xb[j+1]);
                }
        }
};

template<int64_t N>
struct Map2VS< DivOp<TDouble>, N, true > {
        static void eval(State& state, double const* a, double const b, double* r) {
                __m128d const* xa = (__m128d const*)a;
                const __m128d xb = _mm_set1_pd(b);
                __m128d* xr = (__m128d*)r;
                for(int j = 0; j < N/2; j+=2) {
                        xr[j+0] = _mm_div_pd(xa[j+0], xb);
                        xr[j+1] = _mm_div_pd(xa[j+1], xb);
                }
        }
};

#endif
