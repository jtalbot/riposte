
#include <Rinternals.h>
#include <R_ext/GraphicsEngine.h>

void R_GE_checkVersionOrDie(int version) {
    throw "NYI: R_GE_checkVersionOrDie";
}

pGEDevDesc Rf_desc2GEDesc(pDevDesc dd) {
    throw "NYI: Rf_desc2GEDesc";
}

pGEDevDesc GEgetDevice(int) {
    throw "NYI: GEgetDevice";
}

void GEaddDevice(pGEDevDesc) {
    throw "NYI: GEaddDevice";
}

void GEaddDevice2(pGEDevDesc, const char *) {
    throw "NYI: GEaddDevice2";
}

void GEkillDevice(pGEDevDesc) {
    throw "NYI: GEkillDevice";
}

pGEDevDesc GEcreateDevDesc(pDevDesc dev) {
    throw "NYI: GEcreateDevDesc";
}

/* Convert an element of a R colour specification (which might be a
   number or a string) into an internal colour specification. */
rcolor RGBpar(SEXP, int) {
    throw "NYI: RGBpar";
}

/* Convert an internal colour specification to/from a colour name */
const char *col2name(rcolor col) { /* used in par.c, grid */
    throw "NYI: col2name";
}

/* Convert either a name or a #RRGGBB[AA] string to internal.
   Because people were using it, it also converts "1", "2" ...
   to a colour in the palette, and "0" to transparent white.
*/
rcolor R_GE_str2col(const char *s) {
    throw "NYI: R_GE_str2col";
}

SEXP GECap(pGEDevDesc dd) {
    throw "NYI: GECap";
}

/*
 * Raster operations
 */
void R_GE_rasterInterpolate(unsigned int *sraster, int sw, int sh,
                            unsigned int *draster, int dw, int dh) {
    throw "NYI: R_GE_rasterInterpolate";
}

pGEDevDesc GEcurrentDevice(void) {
    throw "NYI: GEcurrentDevice";
}

void GEinitDisplayList(pGEDevDesc dd) {
    throw "NYI: GEinitDisplayList";
}

void GEplayDisplayList(pGEDevDesc dd) {
    throw "NYI: GEplayDisplayList";
}

void GEcopyDisplayList(int fromDevice) {
    throw "NYI: GEcopyDisplayList";
}

SEXP GEcreateSnapshot(pGEDevDesc dd) {
    throw "NYI: GEcreateSnapshot";
}

void GEplaySnapshot(SEXP snapshot, pGEDevDesc dd) {
    throw "NYI: GEplaySnapshot";
}

/* From ../../main/plot.c, used by ../../library/grid/src/grid.c : */
// Also used by grDevices
SEXP Rf_CreateAtVector(double*, double*, int, Rboolean) {
    throw "NYI: CreateAtVector";
}

void Rf_GAxisPars(double *min, double *max, int *n, Rboolean log, int axis) {
    throw "NYI: Rf_GAxisPars";
}


// From main/colors.c, used by grDevices

extern "C" {

typedef unsigned int (*F1)(SEXP x, int i, unsigned int bg);
typedef const char * (*F2)(unsigned int col);
typedef unsigned int (*F3)(const char *s);
typedef void (*F4)(Rboolean save);

static F1 ptr_RGBpar3;
static F2 ptr_col2name;
static F3 ptr_R_GE_str2col;
static F4 ptr_savePalette;

void Rg_set_col_ptrs(F1 f1, F2 f2, F3 f3, F4 f4)
{
    ptr_RGBpar3 = f1;
    ptr_col2name = f2;
    ptr_R_GE_str2col = f3;
    ptr_savePalette = f4;
}

}

// .Internals used by grDevices

extern "C" {

SEXP do_X11(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: do_X11";
}

SEXP do_contourLines(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: do_contourLines";
}

SEXP do_getGraphicsEvent(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: do_getGraphicsEvent";
}

SEXP do_getGraphicsEventEnv(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: do_getGraphicsEventEnv";
}

SEXP do_saveplot(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: do_saveplot";
}

SEXP do_setGraphicsEventEnv(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: do_setGraphicsEventEnv";
}

SEXP do_getSnapshot(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: do_getSnapshot";
}

SEXP do_playSnapshot(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: do_playSnapshot";
}

}

