
#ifndef RIPOSTE_EXCEPTIONS_H
#define RIPOSTE_EXCEPTIONS_H

#include <string>

class RiposteException {
public:
	RiposteException() {}
	virtual std::string what() const = 0;
};

// An R-level error
class RiposteError : public RiposteException {
	std::string message;
public:
	RiposteError(std::string m) : RiposteException(), message(m) {}
	std::string what() const { return message; }
};

class CompileError : public RiposteException {
	std::string message;
public:
	CompileError(std::string m) : RiposteException(), message(m) {}
	std::string what() const { return message; }
};

class RuntimeError : public RiposteException {
	std::string message;
public:
	RuntimeError(std::string m) : RiposteException(), message(m) {}
	std::string what() const { return message; }
};

#define THROW_TO_GDB
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
#define _error(T) do { attachGDB(); throw RiposteError(T); } while(0)
#else
#define _error(T) (throw RiposteError(T))
#endif
#define _warning(S, T) (S.warnings.push_back(T))

#endif
