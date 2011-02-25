/*
 * main.cpp
 *   commandline execution for luaR
 *
 * Drawn from Example of a C program that interfaces with Lua.
 * Based on Lua 5.0 code by Pedro Martelletto in November, 2003.
 * Updated to Lua 5.1. David Manura, January 2007.
 */

#define MATHLIB_STANDALONE
#include <stdlib.h>
#include <stdio.h>
#include <err.h>
#include <fcntl.h>
#include <string>
#include <getopt.h>
#include <fstream>
#include <iostream>
extern "C" {
#include <Rmath.h>
}
#include <valgrind/callgrind.h>

#include "rinst.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#include "value.h"
#include "internal.h"

extern void parse(SEXP s, Value& v);

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
 
/*
  start up an instance of R to use for parsing
*/
int embedR(int argc, char const **argv){
  
  structRstart rp;
  Rstart Rp = &rp;
  if (!getenv("R_HOME")) {
    e_message("FATAL","embedR","R_HOME is not set. Please set all required environment variables before running this program.");
 
       return(-1);
    } 

/* 
   This all looks great, but none of it seems to have any effect
*/
  R_running_as_main_program = 1;
  R_DefParams(Rp);
  // vanilla
  Rp->SaveAction = SA_NOSAVE;
  Rp->RestoreAction = SA_NORESTORE;
  Rp->LoadSiteFile = (Rboolean)FALSE;
  Rp->LoadInitFile = (Rboolean)FALSE;
  R_RestoreHistory = (Rboolean)0;
  Rp->NoRenviron = TRUE;
  Rp->R_Interactive = (Rboolean)0;
  Rp->R_Quiet = (Rboolean)TRUE;
  R_SetParams(Rp);

  int stat= Rf_initialize_R(argc, (char **) argv);
  if (stat<0) {
     char msg[64];
     sprintf(msg,"Failed to initialize embedded R rc=%d",stat);
     e_message("FATAL","InitR",msg); 
    return(-2);
  }

  R_SignalHandlers=0;
  R_CStackLimit = (uintptr_t)-1;

  R_Outputfile = NULL;
  R_Consolefile = NULL;
  R_Interactive = (Rboolean)1;
  ptr_R_ShowMessage = Re_ShowMessage;
  ptr_R_WriteConsoleEx =Re_WriteConsoleEx;

  ptr_R_WriteConsole = NULL;
  ptr_R_ReadConsole = NULL;
  
  return(0);
}

/*
  Read stdin and return R expressions (empty of end-of-file)
*/
SEXP parseR() {
	SEXP cmdSexp, cmdexpr = R_NilValue;
	int i = 0;
	int is_tty = isatty(fileno(stdin));
	char code[8192];
	ParseStatus status = PARSE_INCOMPLETE;
	
	while(status == PARSE_INCOMPLETE)
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
                    return R_NilValue;
                  }
		i = strlen(code);
		if(i == 1)
			i = 0;
		if(i > 0) {		
			PROTECT(cmdSexp = Rf_allocVector(STRSXP, 1));
			SET_STRING_ELT(cmdSexp, 0, Rf_mkChar(code));
			cmdexpr = PROTECT(R_ParseVector(cmdSexp, -1, &status, R_NilValue));
			UNPROTECT(2);
		}
	}
	
        // if not a terminal echo the parse so as to interleave with output
	if (!is_tty) fprintf(stdout,"%s",code); 
	if (status != PARSE_OK) {
		e_message("Error","Rparse", code);
	}
	return cmdexpr;
}

SEXP parseR2(char const* code) {
	SEXP cmdSexp, cmdexpr;
	ParseStatus status;
	
	PROTECT(cmdSexp = Rf_allocVector(STRSXP, 1));
	SET_STRING_ELT(cmdSexp, 0, Rf_mkChar(code));
	cmdexpr = PROTECT(R_ParseVector(cmdSexp, -1, &status, R_NilValue));
	
	if (status != PARSE_OK) {
		char msg[64];
        	sprintf(msg,"ParseStatus=%d",status);
		e_message("Error","R",msg);
                cmdexpr = R_NilValue;
	}
	UNPROTECT(2);
	return cmdexpr;
}

SEXP evalR(SEXP code, SEXP env) {
	int i;
	SEXP ans = R_NilValue;
	PROTECT(code);
	PROTECT(env);
	for(i = 0; i < Rf_length(code); ++i) {
		ans = Rf_eval(VECTOR_ELT(code, i), env);
	}
	UNPROTECT(2);
	return ans;
}


int dostdin(Environment* baseenv) {
	SEXP code;
	int status, rc = 0;

	printf("\n");
	printf("A Quick Riposte!   (Fast R)\n\n");
	printf("Justin Talbot\n");
	printf("Stanford University\n");
	printf("rockit@graphics.stanford.edu\n");
	printf("\n");

	State state(new Stack(), new Environment(baseenv, baseenv), baseenv);
	while(!die) {
        status = 0;
		code = parseR();
        if (R_NilValue == code) continue;

		Value value, result;
		parse(code, value);
		//std::cout << "Parsed: " << value.toString() << std::endl;
		//interpret(value, env, result); 
		Block b = compile(state, value);
		//std::cout << "Compiled code: " << b.toString() << std::endl;
		eval(state, b);
		result = state.stack->pop();	
		std::cout << result.toString() << std::endl;
	}
	return rc;
}

static int dofile(const char * file, Environment* baseenv, bool echo) {
	int rc = 0;
	std::string s;

	// Read in the file ourselves
	std::string code;
	std::string line;
	std::ifstream in(file);
       	d_message(1,"Parsing file (%s)\n",file);
	while(std::getline(in,line))
        {
		if (echo && verbose) l_message(1,line.c_str());
		code += line + '\n';
        }	
	// Get R to parse it, should get parse errors here, rather than a crash during eval
	SEXP expressions = PROTECT(parseR2(code.c_str()));
    if (expressions == R_NilValue) { UNPROTECT(1); return 1; }

	timespec begin;
	get_time(begin);
	State state(new Stack(), new Environment(baseenv, baseenv), baseenv);
	for(int i = 0; i < Rf_length(expressions) && !rc; ++i) {
		Value value, result;
		parse(VECTOR_ELT(expressions, i), value);
		//interpret(value, env, result);
		Block b = compile(state, value);
		std::cout << b.toString() << std::endl;
		eval(state, b);	
		result = state.stack->pop();	
		if(echo) {
			std::cout << result.toString() << std::endl;
		}
	}
	UNPROTECT(1);
	print_time_elapsed("dofile", begin);
	return rc;
}

/*static int luaR_quit() {
    die = 1;
    return 0;
}*/

/*static const luaL_reg vectorlib[] = {
	{"allocVector", luaB_createtable},
	{"dumpLuaCode", luaR_dumpLuaCode},
	{"source", luaR_source},
	{"parse", luaR_parse},
	{"quit", luaR_quit},
	{NULL, NULL}
};*/

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

static struct option longopts[] = {
             { "debug",     0,     NULL,           'd' },
             { "file",      1,     NULL,           'f' },
             { "help",      0,     NULL,           'h' },
             { "jitopts",   1,     NULL,           'j' },
             { "verbose",   0,     NULL,           'v' },
             { NULL,        0,     NULL,            0 }
     };

/*  Parse commandline options  */

while ((ch = getopt_long(argc, argv, "df:hj:v", longopts, NULL)) != -1)
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
             default:
                     usage();
                     exit (-1);
     }

     argc -= optind;
     argv += optind;


      d_message(1,NULL,"Command option processing complete");

/* Create  an R instance  to use for parsing */
        char const *  av[]= {"riposte", "--gui=none", "--no-save"};

	if ((rc = embedR(3,av))) {
                e_message("FATAL","riposte","Unable to instance R");
		exit(rc);
	}

    /* Initialize R  */
	setup_Rmainloop();
	
     d_message(1,NULL,"R instanced");

    /* if specified, pass options to luajit  */
    /*if(NULL  != jitopts ){ 
       d_message(1,"jitopts(%s)\n",jitopts);
       dojitcmd(L, jitopts);    
    }*/

	//printf(">> %d\n", sizeof(Value));
	//printf(">> %d\n", sizeof(Instruction));

	/* start garbage collector */
	GC_INIT();

	CALLGRIND_START_INSTRUMENTATION

	/* Create riposte environment */
	Environment* baseenv = new Environment(0,0);
	addMathOps(baseenv);	

/* Either execute the specified file or read interactively from stdin  */

   /* Load the file containing the script we are going to run */
	if(-1 != fd) {
	  close(fd);    /* will reopen in R for parsing */
          d_message(1,"source(%s)\n",filename);
	  rc = dofile(filename,baseenv,true); 
	} else {
	  rc = dostdin(baseenv);
	}

    /* Session over */

        fflush(stdout);
        fflush(stderr);
    
    /* Clean up R */
    R_RunExitFinalizers();
    Rf_KillAllDevices();
    R_CleanTempDir();
 
	return rc;
}


