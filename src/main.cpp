
#include <stdlib.h>
#include <stdio.h>
#include <err.h>
#include <fcntl.h>
#include <string>
#include <getopt.h>
#include <fstream>
#include <iostream>

#include "value.h"
#include "parser/parser.h"
#include "interpreter.h"
#include "jit.h"
#include "library.h"

void registerCoreFunctions(State& state);
void registerCoerceFunctions(State& state);

extern int opterr;

/*  Globals  */
static int debug = 0;
static int die   = 0;      /* end session */
static int verbose = 0;

/* debug messages */
static void d_message(const int level, const char* ifmt, const char *msg)
{
	char*  fmt;
	if (debug < level) return;
	fprintf(stderr,"DEBUG: ");
	fmt = (char *)ifmt;
	if (!fmt) 
		fmt = (char *)"%s\n";
	fprintf(stderr, fmt, msg);
	fflush(stderr);
}
 
 /* error messages */
 static void e_message(const char *severity,const char *source, const char *msg)
{
	fprintf(stderr, "%s: (%s) ",severity,source);
	fprintf(stderr, "%s\n", msg);
	fflush(stderr);
}
 
 /* verbose  messages */
 static void l_message(const int level, const char *msg)
{
	if (level > verbose) return;
	fprintf(stderr, "%s\n", msg);
	fflush(stderr);
}
 
Value parsetty(State& state) {
	Value ppr = Value::Nil();
	int i = 0;
	int is_tty = isatty(fileno(stdin));
	char code[8192];
	int status = 0;
	
	while(status == 0)
	{
		if(is_tty){
		  if(i == 0)
		   printf("> ");
		  else
		   printf("+ ");
		}
		if (NULL == fgets(code+i, 8192-i, stdin))
                  {
                    // EOF exit
                    //l_message(0,"EOF");
                    //die = 1;
                    return Value::Nil();
                  }
		i = strlen(code);
		if(i == 1)
			i = 0;
		if(i > 0) {
			Parser parser(state);
			status = parser.execute(code, i, true, ppr);	
		}
	}
	
        // if not a terminal echo the parse so as to interleave with output
	if (!is_tty) fprintf(stdout,"%s",code); 
	if (status == -1)
		return Value::Nil();
	return ppr;
}

int dostdin(State& state) {
	int rc = 0;

	std::cout << std::endl;
	std::cout << "Riposte (" << state.threads.size() << " threads) -- A Fast Interpreter and JIT for R" << std::endl;
	std::cout << "Flags: \t-v\t verbose vector output" << std::endl;
	std::cout << "\t-j #\t start with # worker threads" << std::endl;
	std::cout << std::endl;
	std::cout << "Stanford University" << std::endl;
	std::cout << "jtalbot@stanford.edu" << std::endl;
	std::cout << std::endl;

	Thread& thread = state.getMainThread();

	while(!die) {
		try { 
			Value value, result;
			value = parsetty(state);
			if(value.isNil()) continue;
			Code* code = JITCompiler::compile(thread, value);
			result = thread.continueEval(code);
			std::cout << state.stringify(result) << std::endl;
		} catch(RiposteError& error) { 
			e_message("Error", "riposte", error.what().c_str());
		} catch(RuntimeError& error) {
			e_message("Error", "runtime", error.what().c_str());
		} catch(CompileError& error) {
			e_message("Error", "compiler", error.what().c_str());
		} catch(...) {
			e_message("Error", "", "unknown exception thrown");
		}
		if(thread.warnings.size() > 0) {
			std::cout << "There were " << intToStr(thread.warnings.size()) << " warnings." << std::endl;
			for(int64_t i = 0; i < (int64_t)thread.warnings.size(); i++) {
				std::cout << "(" << intToStr(i+1) << ") " << thread.warnings[i] << std::endl;
			}
		}
		thread.warnings.clear();
	}
	return rc;
}

static int dofile(const char * file, std::istream & in, State& state, bool echo) {
	int rc = 0;
	std::string s;

	// Read in the file
	std::string codeStr;
	std::string line;
       	d_message(1,"Parsing file (%s)\n",file);
	while(std::getline(in,line))
        {
		if (echo && verbose) l_message(1,line.c_str());
		codeStr += line + '\n';
        }

	Parser parser(state);
	Value value;
	parser.execute(codeStr.c_str(), codeStr.length(), true, value);	

	if(value.isNil()) return -1;

	try {
		Code* code = JITCompiler::compile(state.getMainThread(), value);
		Value result = state.getMainThread().continueEval(code);
		//if(echo)
			std::cout << state.stringify(result) << std::endl;
	} catch(RiposteError& error) {
		e_message("Error", "riposte", error.what().c_str());
	} catch(RuntimeError& error) {
		e_message("Error", "runtime", error.what().c_str());
	} catch(CompileError& error) {
		e_message("Error", "compiler", error.what().c_str());
	} catch(...) {
		e_message("Error", "", "unknown exception thrown");
	}

	return rc;
}

static void usage()
{
	l_message(0,"usage: riposte [--filename Rscript]");
}

int
main(int argc, char** argv)
{
	int rc;

	/*  getopt parsing  */

	int ch;
	int fd = -1;
	char * filename = NULL;
	bool echo = true;
	int threads = 1; 

	static struct option longopts[] = {
		{ "debug",     0,     NULL,           'd' },
		{ "file",      1,     NULL,           'f' },
		{ "help",      0,     NULL,           'h' },
		{ "verbose",   0,     NULL,           'v' },
		{ "quiet",     0,     NULL,           'q' },
		{ NULL,        0,     NULL,            0 }
	};

	/*  Parse commandline options  */
	opterr = 0;
	while ((ch = getopt_long(argc, argv, "df:hj:vq", longopts, NULL)) != -1)
		switch (ch) {
			case 'd':
				debug++;
				break;
			case 'f':
				filename = optarg;
				if (0 != strcmp("-",filename))
				{
					if ((fd = open(filename, O_RDONLY, 0)) == -1)
						err(1, "unable to open %s", filename);
				}
				break;
			case 'h':
				usage();
				exit (-1);
				break;
			case 'v':
				verbose++;
				break;
			case 'q':
				echo = false;
				break;
			case 'j':
				if(0 != strcmp("-",optarg)) {
					threads = atoi(optarg);
				}
				break;
			default:
				//usage();
				//exit (-1);
				break;
		}

	d_message(1,NULL,"Command option processing complete");

	State state(threads, argc, argv);
	state.verbose = verbose;
	Thread& thread = state.getMainThread();

	try {
		registerCoreFunctions(state);	
		registerCoerceFunctions(state);	
		loadLibrary(thread, "library", "core");
		//loadLibrary(thread, "library", "base");
		//loadLibrary(thread, "library", "stats");

	} catch(RiposteError& error) { 
		e_message("Error", "riposte", error.what().c_str());
	} catch(RuntimeError& error) {
		e_message("Error", "runtime", error.what().c_str());
	} catch(CompileError& error) {
		e_message("Error", "compiler", error.what().c_str());
	}
	if(thread.warnings.size() > 0) {
		std::cout << "There were " << intToStr(thread.warnings.size()) << " warnings." << std::endl;
		for(int64_t i = 0; i < (int64_t)thread.warnings.size(); i++) {
			std::cout << "(" << intToStr(i+1) << ") " << thread.warnings[i] << std::endl;
		}
	}
	thread.warnings.clear();
	/* Either execute the specified file or read interactively from stdin  */

	/* Load the file containing the script we are going to run */
	if(-1 != fd) {
		close(fd);    /* will reopen in R for parsing */
		d_message(1,"source(%s)\n",filename);
		std::ifstream in(filename);
		rc = dofile(filename,in,state,echo);
	} else if(isatty(STDIN_FILENO)){
		rc = dostdin(state);
	} else {
		rc = dofile("<stdin>",std::cin,state,echo);
	}

	/* Session over */

	fflush(stdout);
	fflush(stderr);

	return rc;
}


