
#include <stddef.h>
#include <R_ext/Riconv.h>

void * Riconv_open (const char* tocode, const char* fromcode) {
    throw "NYI: Riconv_open";
}

size_t Riconv (void * cd, const char **inbuf, size_t *inbytesleft,
           char  **outbuf, size_t *outbytesleft) {
    throw "NYI: Riconv";
}

int Riconv_close (void * cd) {
    throw "NYI: Riconv_close";
}

