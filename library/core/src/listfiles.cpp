
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "../../../src/runtime.h"
#include "../../../src/interpreter.h"

extern "C"
void listfiles_map(State& state,
    Value& out,
    Character::Element path)
{
    Character result(0);

    if(path != Character::NAelement) {
        std::vector<std::string> names;
        DIR* dp;
        struct dirent* ep;
        dp = opendir(path->s);
        if(dp != NULL) {
            while((ep = readdir(dp))) {
                names.push_back(ep->d_name);
            }
            closedir(dp);
        }
        else {
            printf("Couldn't open directory: %s\n", path->s);
        }

        result = Character(names.size());
        for(size_t i = 0; i < names.size(); ++i) {
            result[i] = state.internStr(names[i].c_str());
        }
    }

    out = result;
}

