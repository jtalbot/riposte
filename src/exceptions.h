
#ifndef RIPOSTE_EXCEPTIONS_H
#define RIPOSTE_EXCEPTIONS_H

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

#define _error(T) (throw RiposteError(T))
#define _warning(S, T) (S.warnings.push_back(T))

#endif
