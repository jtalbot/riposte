
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

#include "../../../libs/dyncall/dyncall/dyncall.h"

extern "C"
Value dotC(Thread& thread, Value const* args) {
    Externalptr const& func = (Externalptr const&)args[0];
    List const& arguments = (List const&)args[1];

    DCCallVM* vm = dcNewCallVM(4096);
    dcMode(vm, DC_CALL_C_DEFAULT);

    // TODO: need to copy all the arguments, transforming those necessary
    for(size_t i = 0; i < arguments.length(); ++i) {
        switch(arguments[i].type()) {
            case Type::List:
                dcArgPointer(vm, (void*)((List const&)arguments[i]).v());
                break;
            default:
                throw "Unsupported type in dotC";
                break;
        }
    }

    // Make the call using dyncall
    dcCallVoid(vm, func.ptr());
    dcFree(vm);

    // Copy results back out
    List result(arguments.length());
    for(size_t i = 0; i < arguments.length(); ++i) {
        result[i] = arguments[i];
    }

    return result;
}

