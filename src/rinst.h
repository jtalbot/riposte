#ifndef	__rinst_h
#define	__rinst_h


#define R_NO_REMAP
#define R_INTERFACE_PTRS 1
#define CSTACK_DEFNS 1 

#include <Rversion.h>
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <Rinterface.h>
#include <Rembedded.h>
#include <R_ext/Boolean.h>
#include <R_ext/Parse.h>
#include <R_ext/Rdynload.h>

#ifndef DLEVEL
#define DLEVEL 8
#endif

#ifdef RINSTDEBUG
#define LOGG(...) logg(__VA_ARGS__)
#else
#define LOGG(...)
#endif

extern FILE* LOGFP;
void logg(int ,const char *, ...);
SEXP rexpress(const char*, int count);
void Re_ResetConsole(void);
void Re_FlushConsole(void);
void Re_ClearerrConsole(void);
void Re_WriteConsoleEx(const char *, int , int);
void Re_ShowMessage(const char* );
void cleanup_R();
int load_file_to_memory(const char *, char **) ;

extern int R_running_as_main_program;
extern int R_RestoreHistory;
extern uintptr_t R_CStackLimit; 

#endif
