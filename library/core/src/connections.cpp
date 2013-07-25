
#include <iostream>
#include <fstream>
#include <string>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

struct FileConnection {
    std::string name;
    std::fstream f;

    FileConnection(String s)
        : name(s) {}

    ~FileConnection() {
        if(f.is_open()) {
            printf("Automatically closing an unneeded file connection\n");
            f.close();
        }
    }

    void open() {
        f.open(name.c_str());
    }

    void close() {
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
    FileConnection* fc = new FileConnection(desc[0]);
    Value v;
    Externalptr::Init(v, fc, Value::Nil(), Value::Nil(), file_finalize);
    return v;
}

extern "C"
Value file_open(Thread& thread, Value const* args) {
    Externalptr const& p = (Externalptr const&)args[0];
    FileConnection* fc = (FileConnection*)p.ptr();
    fc->open();
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
    fc->write(c[0]);
    return p;
}

extern "C"
Value stdout_cat(Thread& thread, Value const* args) {
    Character const& c = (Character const&)args[0];
    std::cout << c[0];
    return Null::Singleton();
}

extern "C"
Value stderr_cat(Thread& thread, Value const* args) {
    Character const& c = (Character const&)args[0];
    std::cerr << c[0];
    return Null::Singleton();
}

extern "C"
Value file_readLines(Thread& thread, Value const* args) {
    // TODO: implement the other readLines arguments
    // TODO: stream this
    Externalptr const& p = (Externalptr const&)args[0];
    FileConnection* fc = (FileConnection*)p.ptr();
    std::vector<std::string> lines;
    std::string str;
    while(std::getline(fc->f, str)) {
        lines.push_back(str);
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

