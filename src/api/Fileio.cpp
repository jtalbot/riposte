
#include <cstdio>

extern "C" {

// Not exposed by the API, but used by grDevices
FILE *  R_fopen(const char *filename, const char *mode) {
    throw "NYI: R_fopen";
}

}

