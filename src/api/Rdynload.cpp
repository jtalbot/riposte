
#include "api.h"
#include <R_ext/Rdynload.h>

#include "../frontend.h"

Value parseC(R_CMethodDef const* call, String klass) {
    Value n = Character::c(MakeString(call->name));
    Value a;
    Externalptr::Init(a, (void*)(call->fun), Value::Nil(), Value::Nil(), NULL);
    Value d = Null::Singleton();
    Value num = Integer::c(call->numArgs);

    // TODO: parse the types and styles too
    Character names(4);
    names[0] = Strings::name;
    names[1] = Strings::address;
    names[2] = Strings::dll;
    names[3] = Strings::numParameters;

    List r = List::c(n, a, d, num);
    auto dict = new Dictionary(
        Strings::classSym, Character::c(klass, Strings::NativeSymbolInfo),
        Strings::names, names);
    r.attributes(dict);

    return r;
}

Value parseCall(R_CallMethodDef const* call, String klass) {
    Value n = Character::c(MakeString(call->name));
    Value a;
    Externalptr::Init(a, (void*)(call->fun), Value::Nil(), Value::Nil(), NULL);
    Value d = Null::Singleton();
    Value num = Integer::c(call->numArgs);

    // TODO: parse the types and styles too
    Character names(4);
    names[0] = Strings::name;
    names[1] = Strings::address;
    names[2] = Strings::dll;
    names[3] = Strings::numParameters;

    List r = List::c(n, a, d, num);
    auto dict = new Dictionary(
        Strings::classSym, Character::c(klass, Strings::NativeSymbolInfo),
        Strings::names, names);
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
            values.push_back(parseC(a, Strings::CRoutine));
            ++a;
        }
        List r(values.size());
        Character names(values.size());
        for(size_t i = 0; i < values.size(); ++i) {
            r[i] = values[i];
            names[i] = ((List const&)values[i])[0].s;
        }
        auto dict = new Dictionary(
            Strings::classSym, Character::c(Strings::NativeRoutineList),
            Strings::names, names);
        r.attributes(dict);
        args[0] = r;
    }

    if(callRoutines) {
        std::vector<Value> values;
        R_CallMethodDef const* a = callRoutines;
        while(a->name != NULL) {
            values.push_back(parseCall(a, Strings::CallRoutine));
            ++a;
        }
        List r(values.size());
        Character names(values.size());
        for(size_t i = 0; i < values.size(); ++i) {
            r[i] = values[i];
            names[i] = ((List const&)values[i])[0].s;
        }
        auto dict = new Dictionary(
            Strings::classSym, Character::c(Strings::NativeRoutineList),
            Strings::names, names);
        r.attributes(dict);
        args[1] = r;
    }

    if(fortranRoutines) {
        std::vector<Value> values;
        R_CMethodDef const* a = (R_CMethodDef const*)fortranRoutines;
        while(a->name != NULL) {
            values.push_back(parseC(a, Strings::FortranRoutine));
            ++a;
        }
        List r(values.size());
        Character names(values.size());
        for(size_t i = 0; i < values.size(); ++i) {
            r[i] = values[i];
            names[i] = ((List const&)values[i])[0].s;
        }
        auto dict = new Dictionary(
            Strings::classSym, Character::c(Strings::NativeRoutineList),
            Strings::names, names);
        r.attributes(dict);
        args[2] = r;
    }
    
    if(externalRoutines) {
        std::vector<Value> values;
        R_CallMethodDef const* a = externalRoutines;
        while(a->name != NULL) {
            values.push_back(parseCall(a, Strings::ExternalRoutine));
            ++a;
        }
        List r(values.size());
        Character names(values.size());
        for(size_t i = 0; i < values.size(); ++i) {
            r[i] = values[i];
            names[i] = ((List const&)values[i])[0].s;
        }
        auto dict = new Dictionary(
            Strings::classSym, Character::c(Strings::NativeRoutineList),
            Strings::names, names);
        r.attributes(dict);
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
    _NYI("R_FindSymbol");
}

/* Experimental interface for exporting and importing functions from
   one package for use from C code in a package.  The registration
   part probably ought to be integrated with the other registrations.
   The naming of these routines may be less than ideal. */

void R_RegisterCCallable(const char *package, const char *name, DL_FUNC fptr) {
    Externalptr a;
    Externalptr::Init(a, (void*)(fptr), Value::Nil(), Value::Nil(), NULL);
    global->installSEXP(a);
    // TODO: store this in a map where we can look it up later.
}

extern "C" {

// This is in main/Rdynload.c, but is used by grDevices.
int R_cairoCdynload(int local, int now) {
    _NYI("R_cairoCdynload");
}

// From Rdynpriv.h

// Used by methods
SEXP R_MakeExternalPtrFn(DL_FUNC p, SEXP tag, SEXP prot) {
    _NYI("R_MakeExternalPtrFn");
}

}
