
#include "api.h"
#include <R_ext/eventloop.h>

InputHandler *addInputHandler(InputHandler *handlers, int fd, InputHandlerProc handler, int activity) {
    _NYI("addInputHandler");
}

InputHandler *R_InputHandlers;

