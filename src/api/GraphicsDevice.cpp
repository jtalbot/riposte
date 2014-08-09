
#define R_NO_REMAP
#include <Rinternals.h>
#include <R_ext/GraphicsEngine.h>

/* Properly declared version of devNumber */
int Rf_ndevNumber(pDevDesc ) {
    throw "NYI: Rf_ndevNumber";
}

/* Check for an available device slot */
void R_CheckDeviceAvailable(void) {
    throw "NYI: R_CheckDeviceAvailable";
}

/* Return the number of the current device. */
int Rf_curDevice(void) {
    throw "NYI: Rf_curDevice";
}

/* Return the number of the next device. */
int Rf_nextDevice(int) {
    throw "NYI: Rf_nextDevice";
}

/* Return the number of the previous device. */
int Rf_prevDevice(int) {
    throw "NYI: Rf_prevDevice";
}

/* Make the specified device (specified by number) the current device */
int selectDevice(int) {
    throw "NYI: Rf_selectDevice";
}

/* Kill device which is identified by number. */
void Rf_killDevice(int) {
    throw "NYI: Rf_killDevice";
}

int NoDevices(void) { /* used in engine, graphics, plot, grid */
    throw "NYI: Rf_NoDevices";
}

void Rf_NewFrameConfirm(pDevDesc) { /* used in graphics.c, grid */
    throw "NYI: Rf_NewFrameConfirm";
}

