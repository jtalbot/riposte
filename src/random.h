
#ifndef _RIPOSTE_RANDOM_H
#define _RIPOSTE_RANDOM_H

// A simple LCG random number generator that generates independent
// streams for each thread.

struct Random {
	static const uint64_t m = 0x27bb2ee687b0b0fd;
	const uint64_t a;
	uint64_t v;

	static const int64_t primes[100];
	
	Random(uint64_t streamIndex) : a(primes[streamIndex]), v(1) {}

	uint64_t next() {
		v = v * m + a;
		return v;
	}
};

#endif

