
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"
#include "../../../src/api/api.h"
#include "../../../libs/dyncall/dyncall/dyncall.h"

extern "C"
Value dotC(State& state, Value const* args) {
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
Value dotCall(State& state, Value const* args) {
    Externalptr const& func = (Externalptr const&)args[0];
    List const& arguments = (List const&)args[1];

    if(!state.global.apiStack)
        throw "Cannot use .Call interface without R API loaded";

    DCCallVM* vm = dcNewCallVM(4096);
    dcMode(vm, DC_CALL_C_DEFAULT);

    // Get the lock on the global state...
    state.global.apiLock.acquire();
    
    // The API will push user-created SEXPs on the global R_PPStack.
    // Remember the size so we can restore to the correct size.
    int stackSize = *state.global.apiStack->size;

    // Pass as SEXPs, these need to be protected as well...
    // We should probably save them in the state-specific stack instead
    for(size_t i = 0; i < arguments.length(); ++i) {
        SEXP a = new SEXPREC(arguments[i]);
        state.global.apiStack->stack[(*state.global.apiStack->size)++] = a;
        dcArgPointer(vm, (void*)a);
    }

    SEXP result = (SEXP)dcCallPointer(vm, func.ptr());

    (*state.global.apiStack->size) -= arguments.length();
    if(*state.global.apiStack->size != stackSize)
        printf("Protection stack not restored to original size");
    *state.global.apiStack->size = stackSize;

    state.global.apiLock.release();
    
    dcFree(vm);

    return ToRiposteValue(result->v);
}

extern "C"
Value dotExternal(State& state, Value const* args) {
    Externalptr const& func = (Externalptr const&)args[0];
    List const& arguments = (List const&)args[1];

    if(!state.global.apiStack)
        throw "Cannot use .Call interface without R API loaded";

    DCCallVM* vm = dcNewCallVM(4096);
    dcMode(vm, DC_CALL_C_DEFAULT);

    // Get the lock on the global state...
    state.global.apiLock.acquire();
    
    // The API will push user-created SEXPs on the global R_PPStack.
    // Remember the size so we can restore to the correct size.
    int stackSize = *state.global.apiStack->size;

    // Pass as SEXPs, these need to be protected as well...
    // We should probably save them in the state-specific stack instead
    {
        SEXP a = new SEXPREC(arguments);
        state.global.apiStack->stack[(*state.global.apiStack->size)++] = a;
        dcArgPointer(vm, (void*)a);
    }

    SEXP result = (SEXP)dcCallPointer(vm, func.ptr());

    (*state.global.apiStack->size)--;
    if(*state.global.apiStack->size != stackSize)
        printf("Protection stack not restored to original size");
    *state.global.apiStack->size = stackSize;

    state.global.apiLock.release();
    
    dcFree(vm);

    return ToRiposteValue(result->v);
}

