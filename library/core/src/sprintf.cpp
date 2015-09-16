
#include <stdio.h>
#include <stdarg.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

#include "printf-args.c"
#include "printf-parse.c"

#include "../../../libs/dyncall/dyncall/dyncall.h"

extern "C"
void sprintf_map(State& state, String& r, String fmt, Character const* types, ...)
{
    static const size_t maxlength = 8192; // from R documentation
    char out[maxlength];

    DCCallVM* vm = dcNewCallVM(4096);
    dcMode(vm, DC_CALL_C_DEFAULT);
    dcArgPointer(vm, (void*)&out);
    dcArgLongLong(vm, maxlength);
    dcArgPointer(vm, (void*)fmt->s);

    va_list arglist;
    va_start(arglist, types);

    for(int64_t i = 0; i < types->length(); ++i)
    {
        if((*types)[i] == Strings::Logical) {
            char v = va_arg(arglist, int);
            dcArgChar(vm, v);
        }
        else if((*types)[i] == Strings::Integer) {
            int64_t v = va_arg(arglist, int64_t);
            dcArgLongLong(vm, v);
        }
        else if((*types)[i] == Strings::Double) {
            double v = va_arg(arglist, double);
            dcArgDouble(vm, v);
        }
        else if((*types)[i] == Strings::Character) {
            String v = va_arg(arglist, String);
            dcArgPointer(vm, (void*)v->s);
        }
        else if((*types)[i] == Strings::Raw) {
            char v = va_arg(arglist, int);
            dcArgChar(vm, v);
        }
        else if((*types)[i] == Strings::List) {
            _error("Unsupported type in sprintf\n");
        }
    }

    va_end(arglist);

    dcCallVoid(vm, (void*)&snprintf);

    dcFree(vm);

    r = MakeString(out);
}

extern "C"
Value printf_parse(State& state, Value const* args)
{
    auto c = static_cast<Character const&>(args[0]);
    
    Character r(0);

    char_directives d;
    arguments a;

    int success = printf_parse(c.s->s, &d, &a);

    if( success == 0 ) {
        r = Character(a.count);

        size_t j = 0;

        for(size_t i = 0; i < d.count; ++i) {
            if(d.dir[i].width_arg_index != ARG_NONE)
                r[j++] = Strings::Integer;

            if(d.dir[i].precision_arg_index != ARG_NONE)
                r[j++] = Strings::Integer;

            if(d.dir[i].arg_index != ARG_NONE) {
                switch(d.dir[i].conversion) {
                case 'd':
                case 'i':
                case 'u':
                case 'o':
                case 'x':
                case 'X':
                case 'p':
                case 'c':
                    r[j++] = Strings::Integer;
                    break;
                case 'f':
                case 'F':
                case 'e':
                case 'E':
                case 'g':
                case 'G':
                case 'a':
                case 'A':
                    r[j++] = Strings::Double;
                    break;
                case 's':
                    r[j++] = Strings::Character;
                    break;
                default:
                    r[j++] = Strings::NA;
                    break;
                }
            }  
        }
    }

    if (a.arg)
        free (a.arg);
    if (d.dir)
        free (d.dir);

    return r;
}

