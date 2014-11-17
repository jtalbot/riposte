
#include "api.h"
#include <R_ext/Memory.h>

void*   vmaxget(void) {
    // We don't allow the user to control memory management
    return 0;
}

void    vmaxset(const void *) {
    // We don't allow the user to control memory management
}

char*   R_alloc(size_t, int) {
    _NYI("R_alloc");
}

char*   S_alloc(long, int);
char*   S_realloc(char *, long, long, int);


extern "C" {

// The following are in main/memory.c, but is used by grDevices
void *R_chk_calloc(size_t nelem, size_t elsize) {
    // TODO: actually check something here...
    return calloc(nelem, elsize);
}

void R_chk_free(void *ptr) {
    free(ptr);
}

}
