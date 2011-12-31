
#include "library.h"
#include "parser.h"
#include "compiler.h"
#include "value.h"

#include <iostream>
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#include <dlfcn.h>

void sourceFile(State& state, std::string name, Environment* env) {
	//std::cout << "Sourcing " << name << std::endl;
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
		eval(state, Compiler::compile(state, value), env);
	} catch(RiposteError& error) {
		_warning(state, "unable to load library " + name + ": " + error.what().c_str());
	} catch(RuntimeError& error) {
		_warning(state, "unable to load library " + name + ": " + error.what().c_str());
	} catch(CompileError& error) {
		_warning(state, "unable to load library " + name + ": " + error.what().c_str());
	}
}

void openDynamic(State& state, std::string path, Environment* env) {
	std::string p = std::string("/Users/jtalbot/riposte/")+path;
	void* lib = dlopen(p.c_str(), RTLD_LAZY);
	if(lib == NULL) {
		_error(std::string("failed to open: ") + p + " (" + dlerror() + ")");
	}
	// set lib in env...
}

void loadLibrary(State& state, std::string path, std::string name) {
	Environment* env = new Environment(state.sharedState.path.back());
	
	std::string p = path + "/" + name + ("/R/");

	dirent* file;
	struct stat info;

	// Load R files
	DIR* dir = opendir(p.c_str());
	if(dir != NULL) {
		while((file=readdir(dir))) {
			if(file->d_name[0] != '.') {
				stat(file->d_name, &info);
				std::string name = file->d_name;
				if(!S_ISDIR(info.st_mode) && 
						(name.length()>2 && name.substr(name.length()-2,2)==".R")) {
					sourceFile(state, p+name, env);
				}
			}
		}
		closedir(dir);
	}

	// Load dynamic libraries
	p = path + "/" + name + "/libs/";
	dir = opendir(p.c_str());
	if(dir != NULL) {
		while((file=readdir(dir))) {
			if(file->d_name[0] != '.') {
				stat(file->d_name, &info);
				std::string name = file->d_name;
				if(!S_ISDIR(info.st_mode) && 
						(name.length()>2 && name.substr(name.length()-3,3)==".so")) {
					openDynamic(state, p+name, env);
				}
			}
		}
		closedir(dir);
	}

	state.sharedState.path.push_back(env);
	state.sharedState.global->init(state.sharedState.path.back(), 0, Null::Singleton());
}

