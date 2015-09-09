#include <unistd.h>
#include <fstream>
#include <dlfcn.h>

#include "value.h"
#include "bc.h"
#include "interpreter.h"

#include "inst.h"

Global* global;

String MakeString(std::string const& s)
{
    String result = new (s.size()+1) StringImpl();
    memcpy((void*)result->s, s.c_str(), s.size()+1);
    return result;
}

Character InternStrings(State& state, Character const& c)
{
    Character r(c.length());
    for(size_t i = 0; i < c.length(); ++i) {
        r[i] = state.global.strings.intern(c[i]->s);
    }
    return r;
}

void profileStack(State const& state);

//
//    Main interpreter loop 
//
//__attribute__((__noinline__,__noclone__)) 
bool interpret(State& state, Instruction const* pc) {

#ifdef USE_THREADED_INTERPRETER
    #define LABELS_THREADED(name,type,...) (void*)&&name##_label,
    static const void* labels[] = {BYTECODES(LABELS_THREADED)};

    goto *(void*)(labels[pc->bc]);
    #define LABELED_OP(name,type,...)       \
        name##_label: {                     \
            pc = name##_inst(state, *pc);     \
            goto *(void*)(labels[pc->bc]);  \
        } 
    STANDARD_BYTECODES(LABELED_OP)
    done_label: {
        pc = done_inst(state, *pc);
        if(pc != 0)
            goto *(void*)(labels[pc->bc]);
        else
            return true;
    }
    stop_label: { 
        return false; 
    }
#else
    while(pc->bc != ByteCode::done) {
        switch(pc->bc) {
            #define SWITCH_OP(name,type,...) \
            case ByteCode::name: { pc = name##_inst(state, *pc); } break;
            STANDARD_BYTECODES(SWITCH_OP)
            case ByteCode::stop: { 
                return false; 
            } break;
            case ByteCode::done: { 
                pc = done_inst(state, *pc);
                if(pc == 0)
                    return true;
            }
        };
    }
#endif
}

bool profileFlag = false;
void sigalrm_handler(int sig) {
    profileFlag = true;
}

bool profile(State& state, Instruction const* pc) {
#ifdef USE_THREADED_INTERPRETER
    #define LABELS_THREADED(name,type,...) (void*)&&name##_label,
    static const void* labels[] = {BYTECODES(LABELS_THREADED)};

    goto *(void*)(labels[pc->bc]);
    #define LABELED_PROFILE_OP(name,type,...)       \
        name##_label: {                     \
            pc = name##_inst(state, *pc);     \
            if(profileFlag) {               \
                profileStack(state);        \
            }                               \
            goto *(void*)(labels[pc->bc]);  \
        } 
    STANDARD_BYTECODES(LABELED_PROFILE_OP)
    stop_label: { 
        return false; 
    }
    done_label: {
        pc = done_inst(state, *pc);
        if(pc != 0)
            goto *(void*)(labels[pc->bc]);
        else {
            return true;
        }
    }
#else
    while(pc->bc != ByteCode::done) {
        if(profileFlag) {
            profileStack(state);
        }
        switch(pc->bc) {
            #define SWITCH_OP(name,type,...) \
            case ByteCode::name: { pc = name##_inst(state, *pc); } break;
            STANDARD_BYTECODES(SWITCH_OP)
            case ByteCode::stop: { 
                return false; 
            } break;
            case ByteCode::done: { 
                return true;
            }
        };
    }
#endif
}

Value State::evalTopLevel(Code const* code, Environment* environment, int64_t resultSlot) {
    if(global.profile) {
        profileFlag = false;
        signal(SIGALRM, &sigalrm_handler);
        ualarm(1000,1000);
    }

    Value result = eval(code, environment, resultSlot);

    ualarm(0,0);

    return result;
}

Value State::eval(Code const* code) {
    return eval(code, frame.environment, frame.code->registers);
}

Value State::eval(Code const* code, Environment* environment, int64_t resultSlot) {
    uint64_t stackSize = stack.size();
    StackFrame oldFrame = frame;

    // make room for the result
    Instruction const* run = buildStackFrame(*this, environment, code, resultSlot, 0);
    // The first two registers are used for setting a return environment.
    // We just want it in the resultSlot, so set to Nil.
    frame.registers[0] = Value::Nil();
    frame.registers[1] = Value::Nil();

    try {
        bool success = global.profile
            ? profile(*this, run)
            : interpret(*this, run);
        if(success) {
            if(stackSize != stack.size())
                _error("Stack was the wrong size at the end of eval");
            return frame.registers[resultSlot];
        }
        else {
            stack.resize(stackSize);
            frame = oldFrame;
            return Value::Nil();
        }
    } catch(...) {
        stack.resize(stackSize);
        frame = oldFrame;
        throw;
    }
}

Value State::eval(Promise const& p, int64_t resultSlot) {
    
    uint64_t stackSize = stack.size();
    StackFrame oldFrame = frame;
    
    Instruction const* run = force(*this, p, NULL, Value::Nil(), resultSlot, 0);
   
    try {
        bool success = global.profile
            ? profile(*this, run)
            : interpret(*this, run);
        if(success) {
            if(stackSize != stack.size())
                _error("Stack was the wrong size at the end of eval");
            return frame.registers[resultSlot];
        }
        else {
            stack.resize(stackSize);
            frame = oldFrame;
            return Value::Nil();
        }
    } catch(...) {
        stack.resize(stackSize);
        frame = oldFrame;
        throw;
    } 
}


const int64_t Random::primes[100] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
509, 521, 523, 541};

State::State(Global& global, TaskQueue* queue) 
    : global(global)
    , visible(true)
#ifdef EPEE
    , traces(global.epeeEnabled)
#endif
    , random(0)
    , queue(queue)
{
    registers = new Value[DEFAULT_NUM_REGISTERS];
    frame.registers = registers;
    frame.environment = NULL;
    frame.code = NULL;
    frame.isPromise = false;
    frame.returnpc = NULL;
}

Global::Global(uint64_t states, int64_t argc, char** argv) 
    : profile(false)
    , verbose(false)
    , epeeEnabled(true)
    , format(Riposte::RiposteFormat)
    , queues(states) {

    // initialize string table
    #define ENUM_STRING_TABLE(name, str) \
        Strings::name = strings.in(std::string(str), false);
    STRINGS(ENUM_STRING_TABLE);
   
    // initialize basic environments 
    this->empty = new Environment(1,(Environment*)0);
    this->global = new Environment(1,empty);

    this->apiStack = NULL;

    // intialize arguments list
    arguments = Character(argc);
    for(int64_t i = 0; i < argc; i++) {
        arguments[i] = MakeString(std::string(argv[i]));
    }

    promiseCode = new Code();
    Integer bc(2);
    bc[0] = Instruction(ByteCode::force, 2, 0, 2).i;
    bc[1] = Instruction(ByteCode::done, 2, 0, 0).i;
    promiseCode->bc = bc;
    promiseCode->registers = 3;
    promiseCode->expression = Value::Nil();

    symbolDict = Dictionary::Make(Strings::classSym, Character::c(Strings::name));
    callDict = Dictionary::Make(Strings::classSym, Character::c(Strings::call)); 
    exprDict = Dictionary::Make(Strings::classSym, Character::c(Strings::expression));
    pairlistDict = Dictionary::Make(Strings::classSym, Character::c(Strings::pairlist));
    complexDict = Dictionary::Make(Strings::classSym, Character::c(Strings::Complex), Strings::names, Character::c(Strings::Re, Strings::Im));
}

void Code::printByteCode(Global const& global) const {
    std::cout << "Code: " << intToHexStr((int64_t)this) << std::endl;
    std::cout << "\tRegisters: " << registers << std::endl;
    if(constants.length() > 0) {
        std::cout << "\tConstants: " << std::endl;
        for(int64_t i = 0; i < (int64_t)constants.length(); i++)
            std::cout << "\t\t" << i << ":\t" << global.stringify(constants[i]) << std::endl;
    }
    if(bc.length() > 0) {
        std::cout << "\tCode: " << std::endl;
        for(int64_t i = 0; i < (int64_t)bc.length(); i++) {
            Instruction inst(bc[i]);
            std::cout << std::hex << &inst << std::dec << "\t" << i << ":\t" << inst.toString();
            if(inst.bc == ByteCode::call) {
                std::cout << "\t\t(arguments: " << static_cast<CompiledCall const&>(calls[inst.b]).arguments().length() << ")";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

struct ProfileNode {
    int64_t count;
    std::map<std::string, ProfileNode> children;
};
ProfileNode profileRoot;

void profileStack(State const& state) {
    ProfileNode* node = &profileRoot;
    node->count++;
    for(size_t index = 1; index <= state.stack.size(); ++index) {
        StackFrame const& s = (index < state.stack.size())
            ? state.stack[index] : state.frame;

        Value const* v = s.environment->get(Strings::__call__);
        if(!s.isPromise && v && v->isList()) {
            auto l = static_cast<List const&>(*v);
            std::string str = state.deparse(l[0]);
            node = &(node->children[str]);
            node->count++; 
        }
    }
    profileFlag = false;
}

struct Ordered {
    int64_t count;
    std::string str;
};

bool operator<(Ordered const& a, Ordered const& b) {
    return a.count > b.count;
}

void dumpProfile(std::stringstream& out, std::string const& name, int indent,
        ProfileNode const& node, std::map<std::string, int64_t>& leaves) {
    std::vector<Ordered> o;
    int64_t total = 0;
    for(std::map<std::string, ProfileNode>::const_iterator i = 
            node.children.begin(); i != node.children.end(); ++i) {
        Ordered oo;
        oo.count = i->second.count;
        oo.str = i->first;
        o.push_back(oo);
        total += i->second.count;
    }

    if(node.count > 500) {
        for(int j = 0; j < indent; ++j)
            out << " ";
        out << name << ": " << node.count-total << "  ( " << node.count << ")" << std::endl;
    }

    leaves[name] += node.count-total;

    std::sort(o.begin(), o.end());
    for(std::vector<Ordered>::const_iterator i = o.begin(); i != o.end(); ++i) {
        std::map<std::string, ProfileNode>::const_iterator q = node.children.find(i->str);
        dumpProfile(out, q->first, indent+1, q->second, leaves);
    }
}

void Global::dumpProfile(std::string filename) {
    std::stringstream buffer;

    if(profile) {
        std::map<std::string, int64_t> leaves;
        ::dumpProfile(buffer, "root", 0, profileRoot, leaves);
        std::vector<Ordered> o;
        for(std::map<std::string, int64_t>::const_iterator i = leaves.begin();
                i != leaves.end(); ++i) {
            Ordered oo;
            oo.count = i->second;
            oo.str = i->first;
            o.push_back(oo);
        }

        std::sort(o.begin(), o.end());
       
        buffer << "\n\nIndividual functions" << std::endl; 
        for(std::vector<Ordered>::const_iterator i = o.begin(); i != o.end(); ++i)
            buffer << i->str << ": " << i->count << std::endl;
    }

    std::ofstream file(filename.c_str());
    file << buffer.rdbuf() << std::endl;
}

void* Global::get_dl_symbol(std::string const& s)
{
    auto i = dl_symbols.find(s);
    if(i != dl_symbols.end())
        return i->second;

    for(auto& j : dl_handles)
    {
        void* func = dlsym(j.second, s.c_str());
        if(func != nullptr)
        {
            dl_symbols[s] = func;
            return func;
        }
    }

    return nullptr;
}


namespace Riposte {

void initialize(int argc, char** argv,
    int threads, bool verbose, Format format, bool profile) {
    global = new Global(threads, argc, argv);
    global->verbose = verbose;
    global->format = format;
    global->profile = profile;
}

void finalize() {
    delete global;
}

State& newState() {
    // TODO: assign states to different task queues
    ::State* r = new ::State(*global, global->queues.queues[0]);
    global->states.push_back(r);
    return *(State*)r;
}

void deleteState(State& s) {
    for(std::list< ::State* >::reverse_iterator i = global->states.rbegin();
        i != global->states.rend(); ++i) {
        if(*i == (::State*)&s) {
            global->states.erase((++i).base());
            break;
        }
    }
    delete (::State*)&s;
}


std::string getString(State const& state, String str) {
    return std::string(((::String)str)->s);
}

String newString(State const& state, std::string const& str) {
    return (String)((::State&)state).internStr(str);
}


extern "C"
char** environ;

char** getEnv()
{
    return environ;    
}

} // namespace Riposte

