
#include <string>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
void substr_map(State& state,
        Character::Element& result,
        Character::Element text,
        Integer::Element start,
        Integer::Element length) {
    std::string s(text->s);
    result = state.internStr(s.substr(start, length).c_str());
}

extern "C"
void substrassign_map(State& state,
        Character::Element& result,
        Character::Element text,
        Integer::Element start,
        Integer::Element length,
        Character::Element repl) {

    std::string s(text->s);
    result = state.internStr(
        (s.substr(0, start) + repl->s + s.substr(
            std::min((Integer::Element)s.size(),start+length)).c_str()));
}
