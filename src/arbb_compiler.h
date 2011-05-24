#ifndef ARBB_COMPILER_H
#define ARBB_COMPILER_H

#include "value.h"

//same as eval, but returns false if it cannot handle the expression yet
bool arbb_eval(State& state, Closure const& closure, bool verbose);

#endif
