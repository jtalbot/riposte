
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
    Externalptr const& p = (Externalptr const&)v;
    FileConnection* fc = (FileConnection*)p.ptr();
    delete fc;
}

extern "C"
Value file_new(Thread& thread, Value const* args) {
    Character desc = As<Character>(thread, args[0]);
    FileConnection* fc = new FileConnection(desc[0]->s);
    Value v;
    Externalptr::Init(v, fc, Value::Nil(), Value::Nil(), file_finalize);
    return v;
}

extern "C"
Value file_open(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    Character const& mode = (Character const&)args[1];
    FileConnection* fc = (FileConnection*)p.ptr();
    fc->open(std::string(mode[0]->s));
    return p; 
}

extern "C"
Value file_close(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    FileConnection* fc = (FileConnection*)p.ptr();
    fc->close();
    return p; 
}

extern "C"
Value file_cat(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    FileConnection* fc = (FileConnection*)p.ptr();
    Character const& c = (Character const&)args[1];
    fc->write(c[0]->s);
    return p;
}

extern "C"
Value stdout_cat(Thread& thread, Value const* args) {
    Character const& c = (Character const&)args[0];
    std::cout << c[0]->s;
    return Null::Singleton();
}

extern "C"
Value stderr_cat(Thread& thread, Value const* args) {
    Character const& c = (Character const&)args[0];
    std::cerr << c[0]->s;
    return Null::Singleton();
}

extern "C"
Value file_readLines(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    Integer const& n = (Integer const&)args[1];
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
        r[i] = thread.internStr(lines[i].c_str());
    }

    return r;
}

extern "C"
Value file_description(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    FileConnection* fc = (FileConnection*)p.ptr();
    return Character::c(thread.internStr(fc->name));
}

extern "C"
Value gzfile_readLines(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    Integer const& n = (Integer const&)args[1];
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
        r[i] = thread.internStr(lines[i].c_str());
    }

    return r;
}

extern "C"
Value terminal_readLines(Thread& thread, Value const* args) {
    Integer const& n = (Integer const&)args[1];
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
        r[i] = thread.internStr(lines[i].c_str());
    }

    return r;
}

extern "C"
Value file_readBin(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    Integer const& n = (Integer const&)args[1];

    FileConnection* fc = (FileConnection*)p.ptr();
    int64_t length = n[0];

    Raw r(length);
    fc->f.read((char*)r.v(), length);

    return r;
}

extern "C"
Value gzfile_readBin(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    Integer const& n = (Integer const&)args[1];

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

