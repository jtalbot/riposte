
#include <stdlib.h>
#include <stdio.h>
#include <err.h>
#include <fcntl.h>
#include <string>
#include <getopt.h>
#include <fstream>
#include <iostream>

#define MATHLIB_STANDALONE
extern "C" {
#include <Rmath.h>
}

#ifdef USE_CALLGRIND
	#include <valgrind/callgrind.h>
#endif

#include "value.h"
#include "internal.h"
#include "parser.h"

/*  Globals  */
static int debug = 0;
static int die   = 0;      /* end session */
static int verbose = 0;
int dumpLuaCode = 0;

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
	Value ppr = Value::NIL;
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
		   printf(" + ");
		}
		if (NULL == fgets(code+i, 8192-i, stdin))
                  {
                    // EOF exit
                    l_message(0,"EOF");
                    die = 1;
                    return Value::NIL;
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
	if (status == -1) {
		e_message("Error","Rparse", code);
	}
	return ppr;
}

int dostdin(State& state) {
	int rc = 0;

	printf("\n");
	printf("A Quick Riposte!   (Fast R)\n\n");
	printf("Justin Talbot\n");
	printf("Stanford University\n");
	printf("rockit@graphics.stanford.edu\n");
	printf("\n");

	while(!die) {
		Value value, result;
		value = parsetty(state);
		if(value.type == Type::I_nil) continue;
		//std::cout << "Parsed: " << value.toString() << std::endl;
		Block b = compile(state, value);
		//std::cout << "Compiled code: " << b.toString() << std::endl;
		eval(state, b);
		result = state.stack.pop();	
		std::cout << state.stringify(result) << std::endl;
	}
	return rc;
}

static int dofile(const char * file, State& state, bool echo) {
	int rc = 0;
	std::string s;

	// Read in the file
	std::string code;
	std::string line;
	std::ifstream in(file);
       	d_message(1,"Parsing file (%s)\n",file);
	while(std::getline(in,line))
        {
		if (echo && verbose) l_message(1,line.c_str());
		code += line + '\n';
        }

	timespec begin;
	get_time(begin);
	
	Parser parser(state);
	Value ppr;
	parser.execute(code.c_str(), code.length(), true, ppr);	
	
	Expression expressions(ppr);
	for(uint64_t i = 0; i < expressions.length(); i++) {
		Block b = compile(state, expressions[i]);
		eval(state, b);
		Value result = state.stack.pop();
		if(echo)
			std::cout << state.stringify(result) << std::endl;
	}

	print_time_elapsed("dofile", begin);
	return rc;
}

static void usage()
{
  l_message(0,"usage: riposte [--filename Rscript] [--jitopts luajit_options]");
}

int
main(int argc, char** argv)
{
    int rc;
  
/*  getopt parsing  */

int ch;
int fd = -1;
char * filename = NULL;
char * jitopts = NULL;
bool echo = true; 

static struct option longopts[] = {
             { "debug",     0,     NULL,           'd' },
             { "file",      1,     NULL,           'f' },
             { "help",      0,     NULL,           'h' },
             { "jitopts",   1,     NULL,           'j' },
             { "verbose",   0,     NULL,           'v' },
             { "quiet",     0,     NULL,           'q' },
             { NULL,        0,     NULL,            0 }
     };

/*  Parse commandline options  */

while ((ch = getopt_long(argc, argv, "df:hj:vq", longopts, NULL)) != -1)
             switch (ch) {
             case 'd':
                     debug++;
                     break;
             case 'f':
                     if (0 != strcmp("-",filename = optarg))
			{
                     	if ((fd = open(optarg, O_RDONLY, 0)) == -1)
                             err(1, "unable to open %s", filename);
                        }
                     break;
             case 'h':
                     usage();
                     exit (-1);
                     break;
             case 'j':
                     jitopts = optarg;
                     break;
             case 'v':
                     verbose++;
                     break;
             case 'q':
                     echo = false;
                     break;
             default:
                     usage();
                     exit (-1);
     }

     argc -= optind;
     argv += optind;


      d_message(1,NULL,"Command option processing complete");

	//printf(">> %d\n", sizeof(Value));
	//printf(">> %d\n", sizeof(Instruction));

	/* start garbage collector */
	GC_INIT();

#ifdef USE_CALLGRIND
	CALLGRIND_START_INSTRUMENTATION
#endif

	/* Create riposte environment */
	Environment* baseenv = new Environment(0,0);
	State state(new Environment(baseenv, baseenv), baseenv);
	addMathOps(state);	

/* Either execute the specified file or read interactively from stdin  */

   /* Load the file containing the script we are going to run */
	if(-1 != fd) {
	  close(fd);    /* will reopen in R for parsing */
          d_message(1,"source(%s)\n",filename);
	  rc = dofile(filename,state,echo); 
	} else {
	  rc = dostdin(state);
	}

    /* Session over */

        fflush(stdout);
        fflush(stderr);
    
	return rc;
}


