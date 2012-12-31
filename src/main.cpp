/*
    Riposte, Copyright (C) 2010-2012 Stanford University.
    
    main.cpp - the REPL loop
*/

#include <fstream>
#include <unistd.h>
#include <getopt.h>

#include "parser.h"
#include "compiler.h"
#include "library.h"

extern "C" 
{
    #include "../libs/linenoise/linenoise.h"
}

/*  Globals  */
static int debug = 0;
static int verbose = 0;

void registerCoreFunctions(State& state);
void registerCoerceFunctions(State& state);

static void info(State& state, std::ostream& out) 
{
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

// interactive line entry using linenoise
static bool terminal(State& state, std::istream & in, std::ostream & out, Value& code)
{ 
    std::string input;
    code = Value::Nil();

    int status = 0;
    while(status == 0)
    {
        char* line = linenoise( input.size() == 0 ? "> " : "+ " );

        if(line == 0)
            return true;

        input += line;

        if(line[0] != '\0') {
            linenoiseHistoryAdd(line);
            linenoiseHistorySave((char*)".riposte_history");
        }

        free(line);
        
        input += "\n";   /* add the discarded newline back in */
        
        if(input.size() > 0) {
            Parser parser(state);
            status = parser.execute(input.c_str(), input.size(), true, code);    
        }
    }

    if (status == -1)
        code = Value::Nil();

    return false;
}

// piped in from stream 
static bool pipe(State& state, std::istream & in, std::ostream & out, Value& code) 
{
    std::string input;
    code = Value::Nil();

    int status = 0;
    while(!in.eof() && status == 0)
    {
        std::string more;
        std::getline(in, more);
        input += more;

        input += "\n";   /* add the discarded newline back in */
        
        if(input.size() > 0) {
            Parser parser(state);
            status = parser.execute(input.c_str(), input.size(), true, code);    
        }
    }

    if (status == -1)
        code = Value::Nil();

    return in.eof();
}

static void dumpWarnings(Thread& thread, std::ostream& out) 
{
    if(thread.warnings.size() > 0) {
        out << "There were " << intToStr(thread.warnings.size()) << " warnings." << std::endl;
        for(int64_t i = 0; i < (int64_t)thread.warnings.size(); i++) {
            out << "(" << intToStr(i+1) << ") " << thread.warnings[i] << std::endl;
        }
    }
    thread.warnings.clear();
} 

static int run(State& state, std::istream& in, std::ostream& out, bool interactive, bool echo) 
{
    Thread& thread = state.getMainThread();

    int rc = 0;

    if(interactive) 
    {
        linenoiseHistoryLoad((char*)".riposte_history");
    }

    bool done = false;
    while(!done) {
        try { 
            Value code;
            done = interactive ?
                terminal(state, in, out, code) :
                pipe(state, in, out, code);

            if(done || code.isNil()) 
                continue;

            Prototype* proto = Compiler::compileTopLevel(thread, code);
            Value result = thread.eval(proto, state.global);
            
            if(echo)
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
    l_message(0,"options:");
    l_message(0,"    -f, --file         execute R script");
    l_message(0,"    -v, --verbose      enable verbose output");
    l_message(0,"    -j N               launch Riposte with N threads");
}

extern int opterr;

int main(int argc, char** argv)
{
    /*  getopt parsing  */

    static struct option longopts[] = {
        { "debug",     0,     NULL,           'd' },
        { "file",      1,     NULL,           'f' },
        { "help",      0,     NULL,           'h' },
        { "verbose",   0,     NULL,           'v' },
        { "quiet",     0,     NULL,           'q' },
        { "script",    0,     NULL,           's'  },
        { "args",      0,     NULL,           'a'  },
        { NULL,        0,     NULL,            0 }
    };

    /*  Parse commandline options  */
    char * filename = NULL;
    bool echo = true;
    int threads = 1; 

    int ch;
    opterr = 0;
    while ((ch = getopt_long(argc, argv, "df:hj:vqas", longopts, NULL)) != -1)
    {
        // don't parse args past '--args'
        if(ch == 'a')
            break;

        switch (ch) {
            case 'd':
                debug++;
                break;
            case 'f':
                filename = optarg;
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
            case 'h':
            default:
                usage();
                exit(-1);
                break;
        }
    }

    d_message(1,NULL,"Command option processing complete");

    /* Start garbage collector */
    GC_INIT();
    GC_disable();

    /* Initialize execution state */
    State state(threads, argc, argv);
    state.verbose = verbose;
    Thread& thread = state.getMainThread();

    /* Load built in & base functions */
    try {
        registerCoreFunctions(state);   
        registerCoerceFunctions(state); 
        loadLibrary(thread, "library", "core");
    } 
    catch(RiposteException& e) { 
        e_message("Error", e.kind().c_str(), e.what().c_str());
    } 
    dumpWarnings(thread, std::cout);
   
 
    /* Either execute the specified file or read interactively from stdin  */
    int rc;
    if(filename != NULL) {
        std::ifstream in(filename);
        rc = run(state, in, std::cout, false, echo);
    } 
    else {
        info(state, std::cout);
        rc = run(state, std::cin, std::cout, true, echo);
    }


    /* Session over */

    fflush(stdout);
    fflush(stderr);

    return rc;
}

