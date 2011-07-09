
#include "library.h"
#include "parser.h"
#include "compiler.h"
#include "value.h"

#include <iostream>
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>

void sourceFile(State& state, std::string name) {
	try {
		std::ifstream t(name.c_str());
		std::stringstream buffer;
		buffer << t.rdbuf();
		std::string code = buffer.str();

		Parser parser(state);
		Value value;
		FILE* trace = NULL;//fopen((name+"_trace").c_str(), "w");
		parser.execute(code.c_str(), code.length(), true, value, trace);
		//fclose(trace);	
	
		eval(state, Compiler::compile(state, value));
	} catch(RiposteError& error) {
		_warning(state, "Error: unable to load library " + name + ": " + error.what().c_str());
	} catch(RuntimeError& error) {
		_warning(state, "Error: unable to load library " + name + ": " + error.what().c_str());
	} catch(CompileError& error) {
		_warning(state, "Error: unable to load library " + name + ": " + error.what().c_str());
	}
}


void loadLibrary(State& state, std::string library_name) {
	Environment* global = state.global;
	state.global = new Environment(state.path.back());
	
	std::string path = std::string("library/")+library_name+("/R/");

	DIR* dir = opendir(path.c_str());
	dirent* file;
	struct stat info;

	while((file=readdir(dir))) {
		if(file->d_name[0] != '.') {
			stat(file->d_name, &info);
			std::string name = file->d_name;
			if(!S_ISDIR(info.st_mode) && 
				(name.length()>2 && name.substr(name.length()-2,2)==".R")) {
				sourceFile(state, path+name);
			}
		}
	}

	state.path.push_back(state.global);
	std::vector<Symbol> s;
	global->init(state.global, s);
	state.global = global;
}
