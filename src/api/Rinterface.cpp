
#include "api.h"
#define R_NO_REMAP
#include <Rinterface.h>
#include <Rinternals.h>

void R_Suicide(const char *);
char *R_HomeDir(void);
int R_DirtyImage;    /* Current image dirty */
char *R_GUIType;
void R_setupHistory(void);
char *R_HistoryFile; /* Name of the history file */
int R_HistorySize;   /* Size of the history file */
int R_RestoreHistory;    /* restore the history file? */

void onintr(void) {
    _NYI("onintr");
}

void (*ptr_R_Suicide)(const char *);
void (*ptr_R_ShowMessage)(const char *);
int  (*ptr_R_ReadConsole)(const char *, unsigned char *, int, int);
void (*ptr_R_WriteConsole)(const char *, int);
void (*ptr_R_WriteConsoleEx)(const char *, int, int);
void (*ptr_R_ResetConsole)(void);
void (*ptr_R_FlushConsole)(void);
void (*ptr_R_ClearerrConsole)(void);
void (*ptr_R_Busy)(int);
void (*ptr_R_CleanUp)(SA_TYPE, int, int);
int  (*ptr_R_ShowFiles)(int, const char **, const char **,
                   const char *, Rboolean, const char *);
int  (*ptr_R_ChooseFile)(int, char *, int);
int  (*ptr_R_EditFile)(const char *);
void (*ptr_R_loadhistory)(SEXP, SEXP, SEXP, SEXP) = 0;
void (*ptr_R_savehistory)(SEXP, SEXP, SEXP, SEXP) = 0;
void (*ptr_R_addhistory)(SEXP, SEXP, SEXP, SEXP) = 0;

int  (*ptr_R_EditFiles)(int, const char **, const char **, const char *);
// naming follows earlier versions in R.app
SEXP (*ptr_do_selectlist)(SEXP, SEXP, SEXP, SEXP) = 0;
SEXP (*ptr_do_dataentry)(SEXP, SEXP, SEXP, SEXP) = 0;
SEXP (*ptr_do_dataviewer)(SEXP, SEXP, SEXP, SEXP) = 0;
void (*ptr_R_ProcessEvents)();

