
#include <R_ext/RS.h>

void *R_chk_realloc(void *, size_t) {
    throw "NYI: R_chk_realloc";
}
