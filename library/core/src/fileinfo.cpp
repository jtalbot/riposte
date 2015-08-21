
#include <sys/types.h>
#include <sys/stat.h>
#include <grp.h>
#include <pwd.h>
#include <glob.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
void fileinfo_map(State& state,
    double& size, char& isdir, int64_t& mode, 
    int64_t& mtime, int64_t& ctime, int64_t& atime,
    int64_t& uid, int64_t& gid, String& uname, String& grname,
    String f)
{
    struct stat t;
    if( stat(f->s, &t) != -1) {
        size = t.st_size;
        isdir = S_ISDIR(t.st_mode) ? Logical::TrueElement : Logical::FalseElement; 
        mode = t.st_mode % 512;
        mtime = t.st_mtime;
        ctime = t.st_ctime;
        atime = t.st_atime;
        uid = t.st_uid;
        gid = t.st_gid;

        passwd const* p = getpwuid(t.st_uid);
        if(p != NULL)
            uname = MakeString(p->pw_name);
        else
            grname = Character::NAelement;

        group const* g = getgrgid(t.st_gid);        
        if(g != NULL)
            grname = MakeString(g->gr_name);
        else
            grname = Character::NAelement;
    }
    else {
        size = Double::NAelement;
        isdir = Logical::NAelement;
        mode = mtime = ctime = atime = uid = gid = Integer::NAelement;
        uname = grname = Character::NAelement;
    }
}

extern "C"
void pathexpand_map(State& state, String& g, String f)
{
    // glob will expand all magic characters...force it to only expand ~
    if(f != NULL && f->s[0] == '~') {
        std::string file(f->s);
        size_t slash = file.find_first_of('/');
        std::string head = file.substr(0, slash);
      
        glob_t gl;
        if(glob(head.c_str(), GLOB_TILDE | GLOB_NOCHECK, NULL, &gl) == 0) {
            std::string tail = slash == std::string::npos ? "" : file.substr(slash);
            g = MakeString(
                (std::string(gl.gl_pathv[0]) + tail).c_str());
            globfree(&gl);
            return;
        }
    }
    g = f;
}

extern "C"
Value sysglob(State& state, Value const* args)
{
    Character const& c = (Character const&)args[0];
    Logical const& l = (Logical const&)args[1];

    bool dir = Logical::isTrue(l[0]);

    int flags = 0;
    if(dir) 
        flags |= GLOB_MARK;

    glob_t gl;
    gl.gl_pathc = 0;
    gl.gl_pathv = 0;
    for(int64_t j = 0; j < c.length(); ++j) {
        if(c[j] == NULL)
            continue;
        
        glob(c[j]->s, flags, NULL, &gl);
        flags |= GLOB_APPEND;
    }

    Character r(gl.gl_pathc);
    for(size_t i = 0; i < gl.gl_pathc; ++i) {
        r[i] = MakeString(gl.gl_pathv[i]);
    }

    globfree(&gl);
    return r;
}

extern "C"
void realpath_map(State& state, String& g, String f, String winslash)
{
    char* r = realpath(f->s, NULL);
    if(r != NULL) {
        g = MakeString(r);
        free(r);
    }
    else {
        g = Character::NAelement;
    }
}

