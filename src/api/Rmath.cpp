
#include "api.h"
#include <stddef.h>
#include <cmath>

#include <R_ext/Arith.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h>

DL_FUNC  User_norm_fun = NULL;

double R_pow(double x, double y) {
    return pow(x,y);
}

double R_pow_di(double x, int n) {
    return pow(x,n);
}

