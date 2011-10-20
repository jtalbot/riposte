#ifndef _REGISTER_SET_H
#define _REGISTER_SET_H

//bit-string based allocator for registers

typedef uint32_t RegisterSet;
struct RegisterAllocator {
	uint32_t a;
	uint32_t n_registers;
	RegisterAllocator(uint32_t n_reg)
	: a(~0), n_registers(n_reg) {}
	void print() {
		for(int i = 0; i < 32; i++)
			if( a & (1 << i))
				printf("-");
			else
				printf("a");
		printf("\n");
	}
	//try to allocated preferred register
	bool allocate(uint8_t preferred, int8_t * reg) {
		assert(preferred < n_registers);
		if(a & (1 << preferred)) {
			a &= ~(1 << preferred);
			*reg = preferred;
			return true;
		} else return allocate(reg);
	}
	bool allocateWithMask(RegisterSet valid_registers, int8_t * reg) {
		*reg = ffs(a & valid_registers) - 1;
		a &= ~(1 << *reg);
		return (*reg < (int) n_registers);
	}
	bool allocate(int8_t * reg) { return allocateWithMask(~0,reg); }
	RegisterSet live_registers() {
		return a;
	}
	bool is_live(uint8_t reg) {
		return !(a & (1 << reg));
	}
	void clear() {
		a = ~0;
	}
	void free(uint8_t reg) {
		a |= (1 << reg);
	}
};

struct RegisterIterator {
	RegisterIterator(RegisterSet l) {
		live = ~l;
	}
	bool done() { return live == 0; }
	void next() {  live &= ~(1 << value()); }
	uint32_t value() { return ffs(live) - 1; }
private:
	uint32_t live;
};


#endif
