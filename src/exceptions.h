
#ifndef RIPOSTE_EXCEPTIONS_H
#define RIPOSTE_EXCEPTIONS_H

#include <string>

class RiposteException {
public:
	RiposteException() {}
    virtual std::string kind() const = 0;
	virtual std::string what() const = 0;
};

// An R-level error (something is wrong with the users' code)
// Should be propagated through R's error handling mechanism
class RuntimeError : public RiposteException {
	std::string message;
public:
	RuntimeError(std::string m) : RiposteException(), message(m) {}
    std::string kind() const { return "runtime"; }
	std::string what() const { return message; }
};

// A bytecode compiler error (also something is wrong with the users' code)
// Don't propagate through R's error handling mechanism since
// the R level error handling may not compile
class CompileError : public RiposteException {
	std::string message;
public:
	CompileError(std::string m) : RiposteException(), message(m) {}
    std::string kind() const { return "compiler"; }
	std::string what() const { return message; }
};

// An internal error (something is wrong with Riposte)
// Don't propagate, we're hosed.
class InternalError : public RiposteException {
	std::string message;
public:
	InternalError(std::string m) : RiposteException(), message(m) {}
    std::string kind() const { return "internal"; }
	std::string what() const { return message; }
};

//#define THROW_TO_GDB
#ifdef THROW_TO_GDB
#include<stdio.h>
#include<sys/types.h>
#include<sys/wait.h>
#include <unistd.h>
static inline void attachGDB() {
	pid_t pid = getpid();
	pid_t t = fork();
	if(t) {
		int status;
		waitpid(t,&status,0);
	} else {
		char buf[32];
		sprintf(buf,"%d",pid);
		execl("/usr/bin/gdb","/usr/bin/gdb","bin/riposte",buf,NULL);
	}
}
#define _error(T) do { attachGDB(); throw RuntimeError(T); } while(0)
#define _internalError(T) do { attachGDB(); throw InternalError(T); } while(0)
#else
#define _error(T) (throw RuntimeError(T))
#define _internalError(T) (throw InternalError(T))
#endif
#define _warning(S, T) (S.warnings.push_back(T))

#endif
