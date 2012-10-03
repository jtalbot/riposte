
#include <stdlib.h>
#include <stdio.h>
#include <err.h>
#include <fcntl.h>
#include <string>
#include <getopt.h>
#include <fstream>
#include <iostream>

#include "value.h"
#include "parser.h"
#include "compiler.h"
#include "library.h"

void registerCoreFunctions(State& state);
void registerCoerceFunctions(State& state);

extern int opterr;

/*  Globals  */
static int debug = 0;
static int verbose = 0;

static void info(State& state, std::ostream& out) {
	out << "Riposte (" << state.nThreads << " threads) "
        << "-- Copyright (C) 2010-2012 Stanford University" << std::endl;
    out << "http://jtalbot.github.com/riposte/" << std::endl;
}

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
 
static Value parse(State& state, std::istream & in, std::ostream & out, bool interactive) {
    std::string input;
	Value ppr = Value::Nil();
	
	int status = 0;
    while(!in.eof() && status == 0)
    {
        if(interactive) {
            if(input.size() == 0)
                out << "> ";
            else
                out << "+ ";
        }

        std::string more;
        std::getline(in, more);
        input += more;
        input += "\n";   /* add the discarded newline back in */
        
        if(input.size() > 0) {
            Parser parser(state);
            status = parser.execute(input.c_str(), input.size(), true, ppr);	
        }
    }
	
	if (status == -1)
		return Value::Nil();

	return ppr;
}

static void dumpWarnings(Thread& thread, std::ostream& out) {
    if(thread.warnings.size() > 0) {
        out << "There were " << intToStr(thread.warnings.size()) << " warnings." << std::endl;
        for(int64_t i = 0; i < (int64_t)thread.warnings.size(); i++) {
            out << "(" << intToStr(i+1) << ") " << thread.warnings[i] << std::endl;
        }
    }
    thread.warnings.clear();
} 

static int run(State& state, std::istream& in, std::ostream& out, bool interactive) {
	
    Thread& thread = state.getMainThread();

    int rc = 0;

    while(!in.eof()) {
        try { 
            Value code = parse(state, in, out, interactive);

            if(code.isNil()) 
                continue;

            Prototype* proto = Compiler::compileTopLevel(thread, code);
            Value result = thread.eval(proto, state.global);

            out << state.stringify(result) << std::endl;
        } 
        catch(RiposteException& e) { 
            e_message("Error", e.kind().c_str(), e.what().c_str());
        } 
        dumpWarnings(thread, out);
    }
    
    return rc;
}

static void usage()
{
	l_message(0,"usage: riposte [options]... [script [args]...]");
	l_message(0,"Available options are:");
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
    bool jit = true;

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
    while ((ch = getopt_long(argc, argv, "df:hj:vqi", longopts, NULL)) != -1)
        switch (ch) {
            case 'd':
                debug++;
                break;
            case 'f':
                filename = optarg;
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
            case 'i':
                jit = false;
                break;
            default:
                //usage();
                //exit (-1);
                break;
        }

    //argc -= optind;
    //argv += optind;

    d_message(1,NULL,"Command option processing complete");

    /* start garbage collector */
    GC_INIT();

    State state(threads, argc, argv);
    state.verbose = verbose;
    state.jitEnabled = jit;
    Thread& thread = state.getMainThread();

    try {
        registerCoreFunctions(state);	
        registerCoerceFunctions(state);	
        loadLibrary(thread, "library", "core");
        //loadLibrary(thread, "library", "base");
        //loadLibrary(thread, "library", "stats");

    } 
    catch(RiposteException& e) { 
        e_message("Error", e.kind().c_str(), e.what().c_str());
    } 
    dumpWarnings(thread, std::cout);
    
    /* Either execute the specified file or read interactively from stdin  */

    /* Load the file containing the script we are going to run */
    if(filename != NULL) {
        std::ifstream in(filename);
        rc = run(state, in, std::cout, false);
    } 
    else {
        info(state, std::cout);
        rc = run(state, std::cin, std::cout, true);
    }

    /* Session over */

    fflush(stdout);
    fflush(stderr);

    return rc;
}


