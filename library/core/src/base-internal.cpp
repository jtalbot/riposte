
#include <lzma.h>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
Value decompress(Thread& thread, Value const* args) {
    Raw const& data = (Raw const&)args[0];
    Integer const& size = (Integer const&)args[1];
    Character const& comp = (Character const&)args[2];

    int64_t s = size[0];
    char c = comp[0]->s[0];
    
    Raw out(s);
    
    if(c == 'Z') {
        // lzma decompression
        lzma_filter filters[2];
        lzma_options_lzma opts;
        lzma_lzma_preset(&opts, 6);

        filters[0].id = LZMA_FILTER_LZMA2;
        filters[0].options = &opts;
        filters[1].id = LZMA_VLI_UNKNOWN;

        lzma_stream strm = LZMA_STREAM_INIT;
        lzma_ret ret = lzma_raw_decoder(&strm, filters);

        if(ret != LZMA_OK)
            printf("Failed at 1\n");

        strm.next_in = data.v();
        strm.avail_in = data.length();
        strm.next_out = out.v();
        strm.avail_out = s;

        ret = lzma_code(&strm, LZMA_RUN);

        if(ret != LZMA_OK && ret != LZMA_STREAM_END)
            printf("Failed at 2 with %d %d\n", ret, LZMA_STREAM_END);

    }
    else if(c == '1') {
        // gzip decompression
        try {
        boost::iostreams::filtering_istream is;
        is.push(boost::iostreams::zlib_decompressor());
        is.push(boost::iostreams::array_source((const char*)data.v(), data.length()));
        boost::iostreams::read(is, (char*)out.v(), out.length());
        }
        catch( const std::ios_base::failure& e)
        {
                    std::cout << "Caught an ios_base::failure.\n"
                  << "Explanatory string: " << e.what() << '\n'
                  << "Error code: " << e.code() << '\n';
        }
    }
    else {
        printf("Unknown decompression type: %c\n", c);
    }
    
    return out;
}

