
#ifndef JIT_H
#define JIT_H

#include <map>
#include <vector>
#include <set>
#include <tr1/unordered_map>
#include <assert.h>

#include "enum.h"
#include "bc.h"
#include "type.h"
#include "value.h"

#define TRACE_ENUM(_) \
		MAP_BYTECODES(_) \
		FOLD_BYTECODES(_) \
		SCAN_BYTECODES(_) \
		GENERATOR_BYTECODES(_) \
        _(loop, "LOOP", ___) \
        _(jmp, "JMP", ___) \
        _(exit, "exit", ___) \
        _(nest, "nest", ___) \
        _(constant, "const", ___) \
		_(sload, "sload", ___) /* loads are type guarded */ \
        _(curenv, "curenv", ___) \
        _(newenv, "newenv", ___) \
		_(load, "load", ___) /* loads are type guarded */ \
		_(elength, "elength", ___) /* loads are type guarded */ \
		_(slength, "slength", ___) /* loads are type guarded */ \
        _(store, "store", ___) \
        _(sstore, "sstore", ___) \
        _(gproto,   "gproto", ___) \
		_(gtrue, "gtrue", ___) \
		_(gfalse, "gfalse", ___) \
		_(gather, "gather", ___) \
		_(scatter, "scatter", ___) \
		_(phi, "phi", ___) \
        _(length, "LENGTH", ___) \
        _(push, "PUSH", ___) \
        _(pop, "POP", ___) \
        _(strip, "STRIP", ___) \
        _(olength, "olength", ___) \
        _(alength, "alength", ___) \
        _(kill, "kill", ___) \
        _(lenv, "lenv", ___) \
        _(denv, "denv", ___) \
        _(cenv, "cenv", ___) \
        _(brcast, "brcast", ___) \
        _(reshape, "reshape", ___) \
		_(nop, "NOP", ___)

DECLARE_ENUM(TraceOpCode,TRACE_ENUM)

class Thread;

#define SPECIALIZE_LENGTH 16

class JIT {

public:

    enum Arity {
        NULLARY,
        UNARY,
        BINARY,
        TERNARY
    };

    enum Group {
        NOP,
        GENERATOR,
        MAP,
        FILTER,
        FOLD,
        SPLIT
    };

    enum State {
        OFF,
        RECORDING
    };

	typedef size_t (*Ptr)(Thread& thread);

	typedef uint64_t IRRef;

    struct Variable {
        IRRef env;
        int64_t i;

        bool operator<(Variable const& o) const {
            return env < o.env || (env == o.env && i < o.i);
        }
    };

    struct Shape {
        IRRef length;
        bool constant;
        size_t traceLength;

        bool operator==(Shape const& o) const {
            return length == o.length;
        }
        bool operator!=(Shape const& o) const {
            return length != o.length;
        }
        bool operator<(Shape const& o) const {
            return length < o.length;
        }
        static const Shape Empty;
        static const Shape Scalar;
        Shape(IRRef length, bool constant, size_t traceLength) 
            : length(length)
            , constant(constant)
            , traceLength(traceLength) {}
    };

    std::map<size_t, Shape> shapes;

    struct Reenter {
        Instruction const* reenter;
        bool inScope;
    };

	struct IR {
		TraceOpCode::Enum op;
		IRRef a, b, c;

        Type::Enum type;
        Shape in, out;

        double cost;
        short reg;
        bool live;

        bool sunk;

        Reenter reenter;

        void dump() const;

        bool operator==(IR const& o) const {
            return  op == o.op &&
                    a == o.a &&
                    b == o.b &&
                    c == o.c &&
                    type == o.type &&
                    in == o.in &&
                    out == o.out;
        }

        IR()
            : op(TraceOpCode::nop), a(0), b(0), c(0), type(Type::Nil), in(Shape::Empty), out(Shape::Empty), sunk(false) {}
        
        IR(TraceOpCode::Enum op, Type::Enum type, Shape in, Shape out)
            : op(op), a(0), b(0), c(0), type(type), in(in), out(out), sunk(false) {}
        
        IR(TraceOpCode::Enum op, IRRef a, Type::Enum type, Shape in, Shape out)
            : op(op), a(a), b(0), c(0), type(type), in(in), out(out), sunk(false) {}
        
        IR(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum type, Shape in, Shape out)
            : op(op), a(a), b(b), c(0), type(type), in(in), out(out), sunk(false) {}
        
        IR(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum type, Shape in, Shape out)
            : op(op), a(a), b(b), c(c), type(type), in(in), out(out), sunk(false) {}
	};

    struct Phi {
        IRRef a, b;
    };

	unsigned short counters[1024];

	static const unsigned short RECORD_TRIGGER = 4;

    State state;
	Instruction const* startPC;

    std::vector<Value> constants;
    std::map<Value, size_t> constantsMap;
    std::map<size_t, IRRef> uniqueConstants;

    struct StackFrame {
        IRRef environment;
        Prototype const* prototype;
        Instruction const* returnpc;
        Value* returnbase;
        IRRef env;
        int64_t dest;
    };

    std::vector<IR> trace;
    std::vector<IR> code;
    std::vector<bool> fusable;
    std::vector<StackFrame> frames;


    struct Register {
        Type::Enum type;
        Shape shape;

        bool operator<(Register const& o) const {
            return type < o.type ||
                   (type == o.type && shape < o.shape); 
        }
    };

    std::vector<Register> registers;
    std::multimap<Register, size_t> freeRegisters;
    IRRef Loop;
    IRRef TopLevelEnvironment;
    
    struct Trace;
    Trace* rootTrace;
    Trace* dest;

    struct Snapshot {
        std::vector<StackFrame> stack;
        std::map< int64_t, IRRef > slotValues;
        std::map< int64_t, IRRef > slotLengths;
        std::map< int64_t, IRRef > slots;
        std::set<IRRef> memory;
    }; 

	struct Exit {
        Snapshot snapshot;
		Reenter reenter;
        size_t index;
	};
	std::map<size_t, Exit> exits;
    Exit BuildExit(Snapshot const& snapshot, Reenter const& reenter, size_t index); 


	JIT() 
		: state(OFF)
	{
	}

	void start_recording(Instruction const* startPC, Environment const* env, Trace* root, Trace* dest) {
		assert(state == OFF);
		state = RECORDING;
		this->startPC = startPC;
        trace.clear();
        envs.clear();
        constants.clear();
        constantsMap.clear();
        uniqueConstants.clear();
        exits.clear();
        shapes.clear();
        rootTrace = root;
        this->dest = dest;

        // insert empty and scalar shapes.
        constant(Integer::c(0));
        constant(Integer::c(1));

        shapes.insert( std::make_pair( 0, Shape::Empty ) );
        shapes.insert( std::make_pair( 1, Shape::Scalar ) );

        TopLevelEnvironment = insert(trace, TraceOpCode::curenv, 0, 0, 0, Type::Environment, Shape::Empty, Shape::Scalar);
	    envs[env] = TopLevelEnvironment;
    }

    bool record(Thread& thread, Instruction const* pc, bool branch=false) {
        return EmitIR(thread, *pc, branch);
    }

	bool loop(Thread& thread, Instruction const* pc) {
		if(pc == startPC) {
            EmitIR(thread, *pc, false);
            return true;
        }
        else {
            EmitIR(thread, *pc, false);
            return false;
        }
	}

    struct Trace : public gc
    {
        size_t traceIndex;
        static size_t traceCount;
        Trace* root;
        Ptr ptr;
        void* function;
        std::vector<Trace, traceable_allocator<Trace> > exits;
        
        bool InScope;
        Instruction const* Reenter;
        size_t counter;
    };

	void end_recording(Thread& thread);
	
	void fail_recording() {
		assert(state != OFF);
		state = OFF;
	}

	IRRef insert(
        std::vector<IR>& t,
        TraceOpCode::Enum op, 
        IRRef a, 
        IRRef b, 
        IRRef c,
        Type::Enum type, 
        Shape in,
        Shape out);
    
    IR makeConstant(Value const& v);
    
    IRRef duplicate(IR const& ir, std::vector<IRRef> const& forward);
	IRRef load(Thread& thread, int64_t a, Instruction const* reenter);
    IRRef constant(Value const& v);
	IRRef rep(IRRef l, IRRef e, Shape target);
	IRRef recycle(IRRef a, Shape target);
	IRRef cast(IRRef a, Type::Enum type);
    IRRef store(Thread& thread, IRRef a, int64_t c);
	IRRef EmitUnary(TraceOpCode::Enum op, IRRef a, Type::Enum rty, Type::Enum mty);
	IRRef EmitFold(TraceOpCode::Enum op, IRRef a, Type::Enum rty, Type::Enum mty);
	IRRef EmitBinary(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum rty, Type::Enum maty, Type::Enum mbty, Instruction const* inst);
	IRRef EmitTernary(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum rty, Type::Enum maty, Type::Enum mbty, Type::Enum mcty, Instruction const* inst);

    Shape SpecializeLength(size_t length, IRRef irlength);
    Shape SpecializeValue(Value const& v, IR ir);
    Shape MergeShapes(Shape a, Shape b, Instruction const* inst);


    void emitCall(IRRef a, Function const& func, Environment* env, Value const& call, Instruction const* inst) {
        IRRef guard = insert(trace, TraceOpCode::gproto, a, (IRRef)func.prototype(), 0, Type::Function, trace[a].out, Shape::Empty);
        trace[guard].reenter = (Reenter) { inst, true };
        IRRef lenv = insert(trace, TraceOpCode::load, a, 0, 0, Type::Environment, trace[a].out, Shape::Scalar);
        IRRef denv = insert(trace, TraceOpCode::curenv, 0, 0, 0, Type::Environment, Shape::Empty, Shape::Scalar);
        IRRef ne = insert(trace, TraceOpCode::newenv, lenv, denv, constant(call), Type::Environment, Shape::Scalar, Shape::Scalar);
        envs[env] = ne;
    }

    void emitPush(Thread const& thread);

    void storeArg(Environment* env, String name, Value const& v) {
        estore(constant(v), env, name);
    }

    std::map<Environment const*, IRRef> envs;

    IRRef getEnv(Environment* env) {
        std::map<Environment const*, IRRef>::const_iterator i;
        i = envs.find(env);
        if(i != envs.end())
            return i->second;
        else {
            Value v;
            return constant(REnvironment::Init(v, env));
            //_error("Looking up nonexistant environment");
        }
    }

    void estore(IRRef a, Environment* env, String name) {
        Variable v = { getEnv(env), (int64_t)constant(Character::c(name)) };
        trace.push_back(IR(TraceOpCode::store, v.env, v.i, a, Type::Nil, trace[a].out, Shape::Empty));
    }

	void dump(Thread& thread, std::vector<IR> const&);

	void compile(Thread& thread);

	void specialize();
	void schedule();
	void Schedule();

    bool EmitIR(Thread& thread, Instruction const& inst, bool branch);
    bool EmitNest(Thread& thread, Trace* trace);

    IRRef EmitOptIR(Thread& thread, IR ir, std::vector<IR>& code, std::vector<IRRef>& forward, std::tr1::unordered_map<IR, IRRef>& cse, Snapshot& snapshot);
    void Replay(Thread& thread);

    void AssignRegister(size_t index);
    void PreferRegister(size_t index, size_t share);
    void ReleaseRegister(size_t index);
    void AssignSnapshot(Snapshot const& snapshot);
    void RegisterAssignment();
    void RegisterAssign(IRRef i, IR ir);

    IRRef Insert(Thread& thread, std::vector<IR>& code, std::tr1::unordered_map<IR, IRRef>& cse, IR ir);
    IR ConstantFold(Thread& thread, IR ir);
    IR StrengthReduce(IR ir);
    IR Normalize(IR ir);
    enum Aliasing {
        NO_ALIAS,
        MAY_ALIAS,
        MUST_ALIAS
    };
    Aliasing Alias(std::vector<IR> const& code, IRRef i, IRRef j);
    IRRef FWD(std::vector<IR> const& code, IRRef i, bool& loopCarried);
    IRRef DSE(std::vector<IR> const& code, IRRef i, bool& crossedExit);
    IRRef DPE(std::vector<IR> const& code, IRRef i);
    bool Ready(IR ir, std::vector<bool>& done);
    void sink(std::vector<bool>& marks, IRRef i);
    void SINK(void); 
    double Opcost(std::vector<IR>& code, IR ir);
 
	template< template<class X> class Group >
	IRRef EmitUnary(TraceOpCode::Enum op, IRRef a) {
		Type::Enum aty = trace[a].type;

        #define EMIT(TA)                  \
        if(aty == Type::TA)                     \
            return EmitUnary(op, a,             \
                Group<TA>::R::VectorType,    \
                Group<TA>::MA::VectorType);
        UNARY_TYPES(EMIT)
        #undef EMIT
        _error("Unknown type in EmitUnary");
    }

	template< template<class X> class Group >
	IRRef EmitFold(TraceOpCode::Enum op, IRRef a) {
		Type::Enum aty = trace[a].type;

        #define EMIT(TA)                  \
        if(aty == Type::TA)                     \
            return EmitFold(op, a,             \
                Group<TA>::R::VectorType,    \
                Group<TA>::MA::VectorType);
        UNARY_TYPES(EMIT)
        #undef EMIT
        if(aty == Type::Object)
            return 0;
        _error("Unknown type in EmitFold");
    }

	template< template<class X, class Y> class Group >
	IRRef EmitBinary(TraceOpCode::Enum op, IRRef a, IRRef b, Instruction const* inst) {
		Type::Enum aty = trace[a].type;
		Type::Enum bty = trace[b].type;

        #define EMIT(TA, TB)                 \
        if(aty == Type::TA && bty == Type::TB)      \
            return EmitBinary(op, a, b,             \
                Group<TA,TB>::R::VectorType,        \
                Group<TA,TB>::MA::VectorType,       \
                Group<TA,TB>::MB::VectorType,       \
                inst);
        BINARY_TYPES(EMIT)
        #undef EMIT
        if(aty == Type::Object || bty == Type::Object)
            return 0;
        _error("Unknown type pair in EmitBinary");
    }
    
    template< template<class X, class Y, class Z> class Group >
    IRRef EmitTernary(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Instruction const* inst) {
		Type::Enum aty = trace[a].type;
		Type::Enum bty = trace[b].type;
		Type::Enum cty = trace[c].type;

        #define EMIT(TA, TB, TC)                            \
        if(aty == Type::TA && bty == Type::TB && cty == Type::TC)   \
            return EmitTernary(op, a, b, c,                         \
                Group<TA,TB,TC>::R::VectorType,                     \
                Group<TA,TB,TC>::MA::VectorType,                    \
                Group<TA,TB,TC>::MB::VectorType,                    \
                Group<TA,TB,TC>::MC::VectorType,                    \
                inst);
        TERNARY_TYPES(EMIT)
        #undef EMIT
        if(aty == Type::Object || bty == Type::Object || cty == Type::Object)
            return 0;
        _error("Unknown type pair in EmitTernary");
    }
#undef TYPES_TMP

};

namespace std {
    namespace tr1 {
    template<>
        struct hash<JIT::Shape> {
            size_t operator()(JIT::Shape const& key) const {
                return key.length;
            }
        };
    template<>
        struct hash<JIT::IR> {
            hash<JIT::Shape> sh;
            size_t operator()(JIT::IR const& key) const {
                return key.op ^ key.a ^ key.b ^ key.c ^ key.type ^ sh(key.in) ^ sh(key.out);
            }
        };
    }
}

#endif
