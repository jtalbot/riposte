
#include <pthread.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"
#include "../../../src/compiler.h"

struct mapplyargs {
	List const& in;
	List& out;
	Value const& func;
};

void* mapplyheader(void* args, uint64_t start, uint64_t end, State& state) {
	mapplyargs& l = *(mapplyargs*)args;
	List apply(1+l.in.length());
	apply[0] = l.func;
	for(int64_t i = 0; i < l.in.length(); i++)
		apply[i+1] = Value::Nil();
	Code* p = Compiler::compileTopLevel(state, CreateCall(apply));
	return p;
}

void mapplybody(void* args, void* header, uint64_t start, uint64_t end, State& state) {
	mapplyargs& l = *(mapplyargs*)args;
	Code* p = (Code*) header;
	for( size_t i=start; i!=end; ++i ) {
		for(int64_t j=0; j < l.in.length(); j++) {
			Value e;
			Element2(l.in, j, e);
			Value a;
			if(e.isVector())
				Element2(e, i % ((Vector const&)e).length(), a);
			else
				a = e;
			p->calls[0].arguments[j] = a;
		}
		l.out[i] = state.eval(p);
	}
	//return 0;
}

extern "C"
Value mapply(State& state, Value const* args) {
	List const& x = (List const&)args[0];
	Value const& func = args[1];
	// figure out result length
	int64_t len = 1;
	for(int i = 0; i < x.length(); i++) {
		Value e;
		Element2(x, i, e);
		if(e.isVector()) 
			len = (((Vector const&)e).length() == 0 || len == 0) ? 0 : std::max(((Vector const&)e).length(), len);
	}
	List r(len);
	memset(r.v(), 0, len*sizeof(List::Element));
	state.gcStack.push_back(r);

	/*List apply(2);
	apply[0] = func;

	// TODO: should have a way to make a simple function call without compiling,
	// or should have a fast case for compilation
	State istate(state);
	for(int64_t i = 0; i < x.length; i++) {
		apply[1] = x[i];
		r[i] = eval(istate, Compiler::compile(state, CreateCall(apply)));
	}*/

	/*List apply(2);
	apply[0] = func;
	apply[1] = Value::Nil();
	Prototype* p = Compiler::compile(state, CreateCall(apply));
	State istate(state);
	for(int64_t i = 0; i < x.length; i++) {
		p->calls[0].arguments[0] = x[i];
		r[i] = eval(istate, p);
	}*/

	/*pthread_t h1, h2;

	lapplyargs a1 = (lapplyargs) {0, x.length/2, state, x, r, func};
	lapplyargs a2 = (lapplyargs) {x.length/2, x.length, state, x, r, func};

        pthread_create (&h1, NULL, lapplybody, &a1);
        pthread_create (&h2, NULL, lapplybody, &a2);
	pthread_join(h1, NULL);
	pthread_join(h2, NULL);
	*/

	mapplyargs a1 = (mapplyargs) {x, r, func};
	state.queue->doall(state, mapplyheader, mapplybody, &a1, 0, r.length(), 1, 1); 

	state.gcStack.pop_back();
	return r;
}
