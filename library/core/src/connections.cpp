
#include <iostream>
#include <fstream>
#include <string>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

struct FileConnection {
    std::string name;
    std::fstream f;
    boost::iostreams::filtering_istream gz;
    bool isCompressed;

    FileConnection(char const* s)
        : name(s)
        , gz()
        , isCompressed(true) {
        gz.push(boost::iostreams::gzip_decompressor());
        gz.push(f);
    }

    ~FileConnection() {
        if(f.is_open()) {
            printf("Automatically closing an unneeded file connection\n");
            f.close();
        }
    }

    void open(std::string mode) {
        std::ios_base::openmode m = std::ios_base::in;
        if(mode == "w" || mode == "wt")
            m = std::ios_base::out;
        else if(mode == "a" || mode == "at")
            m = std::ios_base::app;
        else if(mode == "rb")
            m = std::ios_base::in | std::ios_base::binary;
        else if(mode == "wb")
            m = std::ios_base::out | std::ios_base::binary;
        else if(mode == "ab")
            m = std::ios_base::app | std::ios_base::binary;

        if(!f.is_open()) {
            f.open(name.c_str(), m);
        }
    }

    void close() {
        if(f.is_open())
            f.close();
    }

    void write(char const* c) {
        f.write(c, strlen(c));
    }
};


extern "C"
void file_finalize(Value v) {
    auto p = static_cast<Externalptr const&>(v);
    FileConnection* fc = (FileConnection*)p.ptr();
    delete fc;
}

extern "C"
Value file_new(State& state, Value const* args) {
    auto desc = As<Character>(args[0]);
    FileConnection* fc = new FileConnection(desc[0]->s);
    Value v;
    Externalptr::Init(v, fc, Value::Nil(), Value::Nil(), file_finalize);
    return v;
}

extern "C"
Value file_open(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    auto mode = static_cast<Character const&>(args[1]);
    FileConnection* fc = (FileConnection*)p.ptr();
    fc->open(std::string(mode[0]->s));
    return p; 
}

extern "C"
Value file_close(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    FileConnection* fc = (FileConnection*)p.ptr();
    fc->close();
    return p; 
}

extern "C"
Value file_cat(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    FileConnection* fc = (FileConnection*)p.ptr();
    auto c = static_cast<Character const&>(args[1]);
    fc->write(c[0]->s);
    return p;
}

extern "C"
Value stdout_cat(State& state, Value const* args) {
    auto c = static_cast<Character const&>(args[0]);
    std::cout << c[0]->s;
    return Null::Singleton();
}

extern "C"
Value stderr_cat(State& state, Value const* args) {
    auto c = static_cast<Character const&>(args[0]);
    std::cerr << c[0]->s;
    return Null::Singleton();
}

extern "C"
Value file_readLines(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    auto n = static_cast<Integer const&>(args[1]);
    FileConnection* fc = (FileConnection*)p.ptr();
    int64_t maxLines = n[0];

    std::vector<std::string> lines;
    std::string str;
    int64_t i = 0;
    while((maxLines < 0 || i < maxLines) && std::getline(fc->f, str)) {
        lines.push_back(str);
        i++;
    }

    Character r(lines.size());
    for(size_t i = 0; i < lines.size(); ++i) {
        r[i] = MakeString(lines[i].c_str());
    }

    return r;
}

extern "C"
Value file_writeLines(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    auto lines = static_cast<Character const&>(args[1]);
    auto sep = static_cast<Character const&>(args[2]);
    FileConnection* fc = (FileConnection*)p.ptr();

    for(int64_t i = 0; i < lines.length(); ++i) {
        fc->f.write(lines[i]->s, strlen(lines[i]->s));
        fc->f.write(sep[0]->s, strlen(sep[0]->s));
    }
    
    return Null::Singleton();
}

extern "C"
Value file_description(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    FileConnection* fc = (FileConnection*)p.ptr();
    return Character::c(MakeString(fc->name));
}

extern "C"
Value gzfile_readLines(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    auto n = static_cast<Integer const&>(args[1]);
    FileConnection* fc = (FileConnection*)p.ptr();
    int64_t maxLines = n[0];

    std::vector<std::string> lines;
    std::string str;
    int64_t i = 0;
   
    if(fc->isCompressed) { 
        while((maxLines < 0 || i < maxLines) && std::getline(fc->gz, str)) {
            lines.push_back(str);
            i++;
        }

        // If we failed to read, perhaps it's not gzip compressed
        // try to read normally...
        if (fc->gz.bad()) {
            fc->f.clear();
            fc->f.seekg(0, std::ios::beg);
            fc->isCompressed = false;
        }
    }

    if(!fc->isCompressed) {
        while((maxLines < 0 || i < maxLines) && std::getline(fc->f, str)) {
            lines.push_back(str);
            i++;
        }
    }

    Character r(lines.size());
    for(size_t i = 0; i < lines.size(); ++i) {
        r[i] = MakeString(lines[i].c_str());
    }

    return r;
}

extern "C"
Value terminal_readLines(State& state, Value const* args) {
    auto n = static_cast<Integer const&>(args[1]);
    int64_t maxLines = n[0];
    
    std::vector<std::string> lines;
    std::string str;
    int64_t i = 0;
    while((maxLines < 0 || i < maxLines) && std::getline(std::cin, str)) {
        lines.push_back(str);
        i++;
    }

    Character r(lines.size());
    for(size_t i = 0; i < lines.size(); ++i) {
        r[i] = MakeString(lines[i].c_str());
    }

    return r;
}

extern "C"
Value terminal_writeLines(State& state, Value const* args) {
    auto lines = static_cast<Character const&>(args[1]);
    auto sep = static_cast<Character const&>(args[2]);

    for(int64_t i = 0; i < lines.length(); ++i) {
        std::cout << lines[i]->s << sep[0]->s;
    }
    
    return Null::Singleton();
}

extern "C"
Value file_readBin(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    auto n = static_cast<Integer const&>(args[1]);

    FileConnection* fc = (FileConnection*)p.ptr();
    int64_t length = n[0];

    Raw r(length);
    fc->f.read((char*)r.v(), length);

    return r;
}

extern "C"
Value gzfile_readBin(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    auto n = static_cast<Integer const&>(args[1]);

    FileConnection* fc = (FileConnection*)p.ptr();
    int64_t length = n[0];

    Raw r(length);

    if(fc->isCompressed) { 
        fc->gz.read((char*)r.v(), length);

        // If we failed to read, perhaps it's not gzip compressed
        // try to read normally...
        if (fc->gz.bad()) {
            fc->f.clear();
            fc->f.seekg(0, std::ios::beg);
            fc->isCompressed = false;
        }
    }

    if(!fc->isCompressed) {
        fc->f.read((char*)r.v(), length);
    }

    return r;
}

extern "C"
Value file_seek(State& state, Value const* args) {
    auto p = static_cast<Externalptr const&>(args[0]);
    auto n = static_cast<Double const&>(args[1]);

    // TODO: implement origin & rw

    FileConnection* fc = (FileConnection*)p.ptr();

    if(!Double::isNA(n[0])) {
        int64_t offset = (int64_t)n[0];
        fc->f.seekg(offset, fc->f.beg);
    }

    int64_t offset = fc->f.tellg();

    return Integer::c(offset);
}

