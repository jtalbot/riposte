
#include "library.h"
#include "parser.h"
#include "compiler.h"
#include "value.h"

#include <iostream>
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#include <dlfcn.h>

void sourceFile(Thread& thread, std::string name, Environment* env) {
	if(thread.state.verbose)
        std::cout << "Sourcing " << name << std::endl;
	try {
		std::ifstream t(name.c_str());
		std::stringstream buffer;
		buffer << t.rdbuf();
		std::string code = buffer.str();

		Parser parser(thread.state);
		Value value;
		FILE* trace = NULL;
		parser.execute(code.c_str(), code.length(), true, value, trace);
		if(!value.isNil())
            thread.eval(Compiler::compileTopLevel(thread, value), env);
    } catch(RiposteException const& e) { 
        _warning(thread, "Unable to load library " + name + " (" + e.kind() + "): " + e.what());
    } 
}

void openDynamic(Thread& thread, std::string path, Environment* env) {
    if(thread.state.verbose)
        std::cout << "Opening dynamic library " << path << std::endl;
    void* lib = dlopen(path.c_str(), RTLD_LAZY|RTLD_LOCAL);
    if(lib == NULL) {
        _error(std::string("failed to open: ") + path + " (" + dlerror() + ")");
    }
    else {
        thread.state.handles[path] = lib;
    }
}

void loadPackage(Thread& thread, std::string path, std::string name) {
	Environment* env = new Environment(1,thread.state.path.back(),0,Null::Singleton());
	
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
					openDynamic(thread, p+name, env);
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
					sourceFile(thread, p+name, env);
				}
			}
		}
		closedir(dir);
	}

	thread.state.path.push_back(env);
	thread.state.global->lexical = env;
    thread.state.namespaces[thread.state.internStr(name)] = env;
}

