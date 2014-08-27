
#ifndef _RIPOSTE_LIBRARY_H
#define _RIPOSTE_LIBRARY_H

#include "value.h"

class State;

void loadPackage(State& state, Environment* env, std::string path, std::string name);

#endif
