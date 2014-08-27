
#include <iconv.h>
#include <vector>

#include "../../../src/frontend.h"

extern "C"
void iconv_map(State& state,
    String& result,
    String str, String from, String to, String sub)
{
    iconv_t descriptor = iconv_open(to->s, from->s);
    
    char* inbuf = (char*)str->s;
    size_t inbytesleft = strlen(str->s);

    char* out = new char[(inbytesleft+1)*4];
    char* outbuf = out;
    size_t outbytesleft = (inbytesleft+1)*4;

    size_t ret = iconv( descriptor, &inbuf, &inbytesleft, &outbuf, &outbytesleft );
    
    if(ret == (size_t) -1) {
        printf("iconv failed\n");
    }
    else {
        result = state.internStr(out);
    }

    iconv_close(descriptor);
}

