
#include <R_ext/Memory.h>

void*   vmaxget(void) {
    throw "NYI: vmaxget";
}

void    vmaxset(const void *) {
    throw "NYI: vmaxset";
}

char*   R_alloc(size_t, int) {
    throw "NYI: R_alloc";
}

char*   S_alloc(long, int);
char*   S_realloc(char *, long, long, int);


extern "C" {

// The following are in main/memory.c, but is used by grDevices
void *R_chk_calloc(size_t nelem, size_t elsize) {
    throw "NYI: R_chk_calloc";
}

void R_chk_free(void *ptr) {
    throw "NYI: R_chk_free";
}

}
