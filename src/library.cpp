
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
	if(state.global.verbose)
        std::cout << "Sourcing " << name << std::endl;
	try {
		std::ifstream t(name.c_str());
		std::stringstream buffer;
		buffer << t.rdbuf();
		std::string code = buffer.str();

		Value value;
		parse(state.global, name.c_str(), 
            code.c_str(), code.length(), true, value);
		
        if(!value.isNil())
            state.eval(Compiler::compileExpression(state, value), env);
    } catch(RiposteException const& e) { 
        _error("Unable to load library " + name + " (" + e.kind() + "): " + e.what());
    } 
}

void openDynamic(State& state, std::string path, Environment* env) {
    if(state.global.verbose)
        std::cout << "Opening dynamic library " << path << std::endl;
    void* lib = dlopen(path.c_str(), RTLD_LAZY|RTLD_LOCAL);
    if(lib == NULL) {
        _error(std::string("failed to open: ") + path + " (" + dlerror() + ")");
    }
    else {
        state.global.dl_handles[path] = lib;
    }
}

void loadPackage(State& state, Environment* env, std::string path, std::string name) {
	
	dirent* file;
	struct stat info;

	// Load dynamic libraries (do this first since they may be used by the R files)
	std::string p = path + "/" + name + "/libs/";
	DIR* dir = opendir(p.c_str());
	if(dir != NULL) {
		while((file=readdir(dir))) {
			if(file->d_name[0] != '.') {
				stat(file->d_name, &info);
				std::string name = file->d_name;
				if(!S_ISDIR(info.st_mode) && 
						(name.length()>5 && name.substr(name.length()-6,6)==".dylib")) {
					openDynamic(state, p+name, env);
				}
			}
		}
		closedir(dir);
	}

	// Load R files
	p = path + "/" + name + ("/R/");
	dir = opendir(p.c_str());
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

	// Load data files
	p = path + "/" + name + ("/data/");
	dir = opendir(p.c_str());
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
}

