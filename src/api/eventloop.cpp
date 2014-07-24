
#include <R_ext/eventloop.h>

InputHandler *addInputHandler(InputHandler *handlers, int fd, InputHandlerProc handler, int activity) {
    throw "NYI: addInputHandler";
}

InputHandler *R_InputHandlers;

