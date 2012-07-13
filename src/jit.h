
#ifndef JIT_H
#define JIT_H

struct Value;
struct Code;
struct Prototype;
class Thread;
class Environment;
struct StackLayout;

struct JITCompiler {

	static Code* compile(Thread& thread, Value const& expr);
	static Code* compile(Thread& thread, Value const& expr, StackLayout* layout);

};

// compilation routines
#endif
