
#include "api.h"
#include <R_ext/Error.h>

void Rf_error(const char * format, ...) {
    printf("Rf_error: ");
    va_list arglist;
    va_start( arglist, format );
    vprintf( format, arglist );
    va_end( arglist );
    printf("\n");
    throw;
}

void    Rf_warning(const char * format, ...) {
    printf("Rf_warning: ");
    va_list arglist;
    va_start( arglist, format );
    vprintf( format, arglist );
    va_end( arglist );
    printf("\n");
}

