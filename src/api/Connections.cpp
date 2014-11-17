
#include "api.h"

// Can't include this because the Rconn struct includes
// a member named 'class' which obviously doesn't work
// well in C++.
//#include <R_ext/Connections.h>
#include <R_ext/Boolean.h>


// The following is actually from RConnections.h which
// isn't part of the public API, but which is used by
// the utils package

typedef void* Rconnection;

extern "C" {

int Rconn_fgetc(Rconnection con) {
   _NYI("Rconn_fgetc"); 
}

int Rconn_printf(Rconnection con, const char *format, ...) {
    _NYI("Rconn_printf");
}

Rconnection getConnection(int n) {
    _NYI("getConnection");
}

void Rf_con_pushback(Rconnection con, Rboolean newLine, char *line) {
    _NYI("RF_con_pushback");
}
}

extern "C" {

// The following are not exposed in header files, but are used by utils
SEXP Rdownload(SEXP call, SEXP op, SEXP args, SEXP env) {
    _NYI("Rdownload");
}

// From main/gzio.h, used by grDevices
typedef void* gzFile;

int R_gzclose (gzFile file) {
    _NYI("R_gzclose");
}

char *R_gzgets(gzFile file, char *buf, int len) {
    _NYI("R_gzgets");
}

gzFile R_gzopen (const char *path, const char *mode) {
    _NYI("R_gzopen");
}

}
