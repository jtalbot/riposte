
#include <stdlib.h>

#include "../../../src/runtime.h"

extern "C"
Value systemfn(State& state, Value const* args)
{
    auto c = static_cast<Character const&>(args[0]);
    auto l = static_cast<Logical const&>(args[1]);

    bool intern = Logical::isTrue(l[0]);

    if(c.length() != 1 || Character::isNA(c[0]))
    {
        if(intern)
            return Character(0);
        else
            return Integer::c(127);
    }

    if(!intern)
    {
        int result = system(c[0]->s);
        return Integer::c(result);
    }
    else
    {
        const int max_buffer = 8095;
        char buffer[max_buffer];

        FILE* stream = popen(c[0]->s, "r");
        std::vector<std::string> lines;
        if(stream) {
            while(!feof(stream))
                if(fgets(buffer, max_buffer, stream) != nullptr)
                    lines.push_back(std::string(buffer, max_buffer));
            pclose(stream);
        }

        Character result(lines.size());
        for(size_t i = 0; i < lines.size(); ++i)
            result[i] = MakeString(lines[i]);

        return result;
    }
}

