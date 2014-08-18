
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

extern "C"
Value dotCall(Thread& thread, Value const* args) {
    Externalptr const& func = (Externalptr const&)args[0];
    List const& arguments = (List const&)args[1];

    if(!thread.state.apiStack)
        throw "Cannot use .Call interface without R API loaded";

    // The API will push user-created SEXPs on the global R_PPStack.
    // Remember the size so we can restore to the correct size.
    int stackSize = *thread.state.apiStack->size;

    DCCallVM* vm = dcNewCallVM(4096);
    dcMode(vm, DC_CALL_C_DEFAULT);

    // Get the lock on the global state...
    thread.state.apiLock.acquire();
    
    // Pass as SEXPs, these need to be protected as well...
    // We should probably save them in the thread-specific stack instead
    for(size_t i = 0; i < arguments.length(); ++i) {
        SEXP a = new SEXPREC(arguments[i]);
        thread.state.apiStack->stack[(*thread.state.apiStack->size)++] = a;
        dcArgPointer(vm, (void*)a);
    }

    SEXP result = (SEXP)dcCallPointer(vm, func.ptr());

    (*thread.state.apiStack->size) -= arguments.length();
    if(*thread.state.apiStack->size != stackSize)
        printf("Protection stack not restored to original size");
    *thread.state.apiStack->size = stackSize;

    thread.state.apiLock.release();
    
    dcFree(vm);

    return result->getValue();
}

extern "C"
Value dotExternal(Thread& thread, Value const* args) {
    Externalptr const& func = (Externalptr const&)args[0];
    List const& arguments = (List const&)args[1];

    if(!thread.state.apiStack)
        throw "Cannot use .Call interface without R API loaded";

    // The API will push user-created SEXPs on the global R_PPStack.
    // Remember the size so we can restore to the correct size.
    int stackSize = *thread.state.apiStack->size;

    DCCallVM* vm = dcNewCallVM(4096);
    dcMode(vm, DC_CALL_C_DEFAULT);

    // Get the lock on the global state...
    thread.state.apiLock.acquire();
    // Pass as SEXPs, these need to be protected as well...
    // We should probably save them in the thread-specific stack instead
    {
        SEXP a = new SEXPREC(arguments);
        thread.state.apiStack->stack[(*thread.state.apiStack->size)++] = a;
        dcArgPointer(vm, (void*)a);
    }

    SEXP result = (SEXP)dcCallPointer(vm, func.ptr());

    (*thread.state.apiStack->size) -= arguments.length();
    if(*thread.state.apiStack->size != stackSize)
        printf("Protection stack not restored to original size");
    *thread.state.apiStack->size = stackSize;

    thread.state.apiLock.release();
    
    dcFree(vm);

    return result->getValue();
}

