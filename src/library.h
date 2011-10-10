
#ifndef _RIPOSTE_LIBRARY_H
#define _RIPOSTE_LIBRARY_H

#include "value.h"

void loadLibrary(State& state, std::string path, std::string name);


void importCoreFunctions(State& state, Environment* env);
void importCoerceFunctions(State& state, Environment* env);


#endif
