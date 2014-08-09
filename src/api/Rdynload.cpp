
#include <R_ext/Rdynload.h>

#include "../frontend.h"

Value parseC(R_CMethodDef const* call, std::string klass) {
    Value n = Character::c(globalState->internStr(call->name));
    Value a;
    Externalptr::Init(a, (void*)(call->fun), Value::Nil(), Value::Nil(), NULL);
    Value d = Null::Singleton();
    Value num = Integer::c(call->numArgs);

    // TODO: parse the types and styles too
    Character names(4);
    names[0] = globalState->internStr("name");
    names[1] = globalState->internStr("address");
    names[2] = globalState->internStr("dll");
    names[3] = globalState->internStr("numParameters");

    List r = List::c(n, a, d, num);
    Dictionary* dict = new Dictionary(2);
    dict->insert(Strings::classSym) =
        Character::c(
            globalState->internStr(klass),
            globalState->internStr("NativeSymbolInfo"));
    dict->insert(Strings::names) = names;
    r.attributes(dict);

    return r;
}

Value parseCall(R_CallMethodDef const* call, std::string klass) {
    Value n = Character::c(globalState->internStr(call->name));

    Value a;
    Externalptr::Init(a, (void*)(call->fun), Value::Nil(), Value::Nil(), NULL);
    Value d = Null::Singleton();
    Value num = Integer::c(call->numArgs);

    // TODO: parse the types and styles too
    Character names(4);
    names[0] = globalState->internStr("name");
    names[1] = globalState->internStr("address");
    names[2] = globalState->internStr("dll");
    names[3] = globalState->internStr("numParameters");

    List r = List::c(n, a, d, num);
    Dictionary* dict = new Dictionary(2);
    dict->insert(Strings::classSym) =
        Character::c(
            globalState->internStr(klass),
            globalState->internStr("NativeSymbolInfo"));
    dict->insert(Strings::names) = names;
    r.attributes(dict);

    return r;
}

int R_registerRoutines(DllInfo *info, const R_CMethodDef * const croutines,
               const R_CallMethodDef * const callRoutines,
               const R_FortranMethodDef * const fortranRoutines,
                       const R_ExternalMethodDef * const externalRoutines) {

    // For now we're actually passing a pointer to an R value
    Value* args = (Value*) info;

    if(croutines) {
        std::vector<Value> values;
        R_CMethodDef const* a = croutines;
        while(a->name != NULL) {
            values.push_back(parseC(a, "CRoutine"));
            ++a;
        }
        List r(values.size());
        Character names(values.size());
        for(size_t i = 0; i < values.size(); ++i) {
            r[i] = values[i];
            names[i] = ((List const&)values[i])[0].s;
        }
        Dictionary* d = new Dictionary(2);
        d->insert(Strings::classSym) =
            Character::c(globalState->internStr("NativeRoutineList"));
        d->insert(Strings::names) = names;
        r.attributes(d);
        args[0] = r;
    }

    if(callRoutines) {
        std::vector<Value> values;
        R_CallMethodDef const* a = callRoutines;
        while(a->name != NULL) {
            values.push_back(parseCall(a, "CallRoutine"));
            ++a;
        }
        List r(values.size());
        Character names(values.size());
        for(size_t i = 0; i < values.size(); ++i) {
            r[i] = values[i];
            names[i] = ((List const&)values[i])[0].s;
        }
        Dictionary* d = new Dictionary(2);
        d->insert(Strings::classSym) =
            Character::c(globalState->internStr("NativeRoutineList"));
        d->insert(Strings::names) = names;
        r.attributes(d);
        args[1] = r;
    }

    if(fortranRoutines) {
        std::vector<Value> values;
        R_CMethodDef const* a = (R_CMethodDef const*)fortranRoutines;
        while(a->name != NULL) {
            values.push_back(parseC(a, "FortranRoutine"));
            ++a;
        }
        List r(values.size());
        Character names(values.size());
        for(size_t i = 0; i < values.size(); ++i) {
            r[i] = values[i];
            names[i] = ((List const&)values[i])[0].s;
        }
        Dictionary* d = new Dictionary(2);
        d->insert(Strings::classSym) =
            Character::c(globalState->internStr("NativeRoutineList"));
        d->insert(Strings::names) = names;
        r.attributes(d);
        args[2] = r;
    }
    
    if(externalRoutines) {
        std::vector<Value> values;
        R_CallMethodDef const* a = externalRoutines;
        while(a->name != NULL) {
            values.push_back(parseCall(a, "ExternalRoutine"));
            ++a;
        }
        List r(values.size());
        Character names(values.size());
        for(size_t i = 0; i < values.size(); ++i) {
            r[i] = values[i];
            names[i] = ((List const&)values[i])[0].s;
        }
        Dictionary* d = new Dictionary(2);
        d->insert(Strings::classSym) =
            Character::c(globalState->internStr("NativeRoutineList"));
        d->insert(Strings::names) = names;
        r.attributes(d);
        args[3] = r;
    }

    return 0;
}

Rboolean R_useDynamicSymbols(DllInfo *info, Rboolean value) {
    Value* args = (Value*) info;
    args[4] = Logical::c(value == TRUE ? Logical::TrueElement : Logical::FalseElement);
    return value;
}

Rboolean R_forceSymbols(DllInfo *info, Rboolean value) {
    Value* args = (Value*) info;
    args[5] = Logical::c(value == TRUE ? Logical::TrueElement : Logical::FalseElement);
    return value;
}

DL_FUNC R_FindSymbol(char const *, char const *,
                       R_RegisteredNativeSymbol *symbol) {
    throw "NYI: R_FindSymbol";
}

/* Experimental interface for exporting and importing functions from
   one package for use from C code in a package.  The registration
   part probably ought to be integrated with the other registrations.
   The naming of these routines may be less than ideal. */

void R_RegisterCCallable(const char *package, const char *name, DL_FUNC fptr) {
    printf("R_RegisterCCallable not implemented. Called on %s: %s\n", package, name); 
}

extern "C" {

// This is in main/Rdynload.c, but is used by grDevices.
int R_cairoCdynload(int local, int now) {
    throw "NYI: R_cairoCdynload";
}

// From Rdynpriv.h

// Used by methods
SEXP R_MakeExternalPtrFn(DL_FUNC p, SEXP tag, SEXP prot) {
    throw "NYI: R_MakeExternalPtrFn";
}

}
