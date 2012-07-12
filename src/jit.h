
#ifndef JIT_H
#define JIT_H

struct Value;
class Environment;
class Thread;
struct Prototype;

struct JITCompiler {

	static Prototype* compileTopLevel(Thread& thread, Environment* env, Value const& expr);
	static Prototype* compileFunctionBody(Thread& thread, Environment* env, Value const& expr);
	static Prototype* compilePromise(Thread& thread, Environment* env, Value const& expr);

};

// compilation routines
#endif
