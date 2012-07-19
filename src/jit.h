
#ifndef JIT_H
#define JIT_H

struct Value;
struct Code;
struct Prototype;
class Thread;
class Environment;

struct JITCompiler {

	static Code* compile(Thread& thread, Value const& expr);
	static Code* compile(Thread& thread, Value const& expr, Environment* env);

};

// compilation routines
#endif
