
#include <R_ext/Applic.h>

void Rdqags(integr_fn f, void *ex, double *a, double *b,
        double *epsabs, double *epsrel,
        double *result, double *abserr, int *neval, int *ier,
        int *limit, int *lenw, int *last, int *iwork, double *work) {
    throw "NYI: Rdqags";
}

void Rdqagi(integr_fn f, void *ex, double *bound, int *inf,
        double *epsabs, double *epsrel,
        double *result, double *abserr, int *neval, int *ier,
        int *limit, int *lenw, int *last,
        int *iwork, double *work) {
    throw "NYI: Rdqagi";
}

void vmmin(int n, double *b, double *Fmin,
       optimfn fn, optimgr gr, int maxit, int trace,
       int *mask, double abstol, double reltol, int nREPORT,
       void *ex, int *fncount, int *grcount, int *fail) {
    throw "NYI: vmmin";
}

void nmmin(int n, double *Bvec, double *X, double *Fmin, optimfn fn,
       int *fail, double abstol, double intol, void *ex,
       double alpha, double bet, double gamm, int trace,
       int *fncount, int maxit) {
    throw "NYI: nmmin";
}

void cgmin(int n, double *Bvec, double *X, double *Fmin,
       optimfn fn, optimgr gr,
       int *fail, double abstol, double intol, void *ex,
       int type, int trace, int *fncount, int *grcount, int maxit) {
    throw "NYI: cgmin";
}

void lbfgsb(int n, int m, double *x, double *l, double *u, int *nbd,
        double *Fmin, optimfn fn, optimgr gr, int *fail, void *ex,
        double factr, double pgtol, int *fncount, int *grcount,
        int maxit, char *msg, int trace, int nREPORT) {
    throw "NYI: lbfgsb";
}

void samin(int n, double *pb, double *yb, optimfn fn, int maxit,
       int tmax, double ti, int trace, void *ex) {
    throw "NYI: samin";
}

/* ------------------ Entry points NOT in the R API --------------- */

/* The following are registered for use in .C/.Fortran */

/* appl/dqrutl.f: interfaces to dqrsl */
void dqrqty_(double *x, int *n, int *k, double *qraux,
              double *y, int *ny, double *qty) {
    throw "NYI: dqrqty";
}

void dqrqy_(double *x, int *n, int *k, double *qraux,
             double *y, int *ny, double *qy) {
    throw "NYI: dqrqy";
}

void dqrcf_(double *x, int *n, int *k, double *qraux,
             double *y, int *ny, double *b, int *info) {
    throw "NYI: dqrcf";
}

void dqrrsd_(double *x, int *n, int *k, double *qraux,
             double *y, int *ny, double *rsd) {
    throw "NYI: dqrrsd";
}

void dqrxb_(double *x, int *n, int *k, double *qraux,
             double *y, int *ny, double *xb) {
    throw "NYI: dqrxb";
}

/* end of registered */

/* hidden, for use in R.bin/R.dll/libR.so */


/* For use in package stats */

/* appl/uncmin.c : */

void fdhess(int n, double *x, double fval, fcn_p fun, void *state,
        double *h, int nfd, double *step, double *f, int ndigit,
        double *typx) {
    throw "NYI: fdhess";
}

/* Also used in packages nlme, pcaPP */
void optif9(int nr, int n, double *x,
        fcn_p fcn, fcn_p d1fcn, d2fcn_p d2fcn,
        void *state, double *typsiz, double fscale, int method,
        int iexp, int *msg, int ndigit, int itnlim, int iagflg,
        int iahflg, double dlt, double gradtl, double stepmx,
        double steptl, double *xpls, double *fpls, double *gpls,
        int *itrmcd, double *a, double *wrk, int *itncnt) {
    throw "NYI: optif9";
}

/* find qr decomposition, dqrdc2() is basis of R's qr(), 
   also used by nlme and many other packages. */
void dqrdc2_(double *x, int *ldx, int *n, int *p,
              double *tol, int *rank,
              double *qraux, int *pivot, double *work) {
    throw "NYI: dqrdc2";
}

void dqrls_(double *x, int *n, int *p, double *y, int *ny,
             double *tol, double *b, double *rsd,
             double *qty, int *k,
             int *jpvt, double *qraux, double *work) {
    throw "NYI: dqrls";
}

