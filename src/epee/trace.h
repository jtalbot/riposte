
#ifndef TRACE_H
#define TRACE_H

#include "../value.h"
#include "../opgroups.h"

#define TRACE_MAX_VECTOR_REGISTERS (32)
#define TRACE_VECTOR_WIDTH (2048)
//maximum number of instructions to record before dropping out of the
//recording interpreter
#define TRACE_MAX_RECORDED (1024)

struct TraceCodeBuffer;
class Trace {

	public:	

		std::vector<IRNode> nodes;
		std::set<Environment*> liveEnvironments;

		struct Output {
			enum Type { REG, MEMORY };
			Type type;
			union {
				Value* reg;
				Environment::Pointer pointer;
			};
			IRef ref;	   //location of the associated store
		};

		std::vector<Output> outputs;

		TraceCodeBuffer * code_buffer;

		size_t n_recorded_since_last_exec;

		int64_t Size;

		Trace();

        std::string stringify() const;

		IRef EmitCoerce(IRef a, Type::Enum dst_type);
		IRef EmitUnary(IROpCode::Enum op, Type::Enum type, IRef a, int64_t data); 
		IRef EmitBinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, int64_t data);
		IRef EmitTrinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, IRef c);
		IRef EmitIfElse(IRef a, IRef b, IRef cond);

		IRef EmitFilter(IRef a, IRef b);
		IRef EmitSplit(IRef x, IRef f, int64_t levels);

		IRef EmitGenerator(IROpCode::Enum op, Type::Enum type, int64_t length, int64_t a, int64_t b);
		IRef EmitRandom(int64_t length);
		IRef EmitIndex(int64_t length, int64_t a, int64_t b);
		IRef EmitSequence(int64_t length, int64_t a, int64_t b);
		IRef EmitSequence(int64_t length, double a, double b);
		IRef EmitConstant(Type::Enum type, int64_t length, int64_t c);
		IRef EmitGather(Vector const& v, IRef i);
		IRef EmitLoad(Vector const& v, int64_t length, int64_t offset);
		IRef EmitSLoad(Vector const& v);
		IRef EmitSStore(IRef ref, int64_t index, IRef value);

        IRef GetRef(Value const& v) {
            if(v.isFuture()) return ((Future const&)v).ref();
            else if(v.isVector()) {
                Vector const& vec = (Vector const&)v;
                if(vec.isScalar()) return EmitConstant(vec.type(), 1, vec.i);
                else return EmitLoad(vec,vec.length(),0);
            }
            _error("GetRef on invalid type");
        }

		void Execute(Thread & thread);
		void Execute(Thread & thread, IRef ref);
		void Reset();

	private:
		void WriteOutputs(Thread & thread);
		std::string toString(Thread & thread);

		void Interpret(Thread & thread);
		void Optimize(Thread& thread);
		void JIT(Thread & thread);

		void MarkLiveOutputs(Thread& thread);
		void SimplifyOps(Thread& thread);
		void AlgebraicSimplification(Thread& thread);
		void CSEElimination(Thread& thread);
		void UsePropogation(Thread& thread);
		void DefPropogation(Thread& thread);
		void DeadCodeElimination(Thread& thread);
		void PropogateShape(IRNode::Shape shape, IRNode& node);
		void ShapePropogation(Thread& thread);
};

class Traces {
    private:
        std::vector<Trace*> availableTraces;
        std::map< int64_t, Trace*> traces;
        bool const& enabled;    // reference to global enabled state

    public:

        Traces(bool const& enabled) : enabled(enabled) {}

        Type::Enum futureType(Value const& v) {
            if(v.isFuture()) 
                return ((Future const&)v).trace()->nodes[((Future const&)v).ref()].type;
            else 
                return v.type();
        }

        IRNode::Shape futureShape(Value const& v) const {
            if(v.isFuture()) {
                return ((Future const&)v).trace()->nodes[((Future const&)v).ref()].outShape;
            }
            else 
                return (IRNode::Shape) { ((Vector const&)v).length(), -1, 1, -1 };
        }

        Trace* getTrace(int64_t length) {
            if(traces.find(length) == traces.end()) {
                if(availableTraces.size() == 0) {
                    Trace* t = new Trace();
                    availableTraces.push_back(t);
                }
                Trace* t = availableTraces.back();
                t->Reset();
                t->Size = length;
                traces[length] = t;
                availableTraces.pop_back();
            }
            return traces[length];
        }

        Trace* getTrace(Value const& a) {
            return getTrace(futureShape(a).length);
        }

        Trace* getTrace(Value const& a, Value const& b) {
            int64_t la = futureShape(a).length;
            int64_t lb = futureShape(b).length;
            if(la == lb || la == 1)
                return getTrace(lb);
            else if(lb == 1)
                return getTrace(la);
            else
                _error("Shouldn't get here");
        }

        Trace* getTrace(Value const& a, Value const& b, Value const& c) {
            int64_t la = futureShape(a).length;
            int64_t lb = futureShape(b).length;
            int64_t lc = futureShape(c).length;
            if(la != 1)
                return getTrace(la);
            else if(lb != 1)
                return getTrace(lb);
            else if(lc != 1)
                return getTrace(lc);
            else
                _error("Shouldn't get here");
        }

        template< template<class X> class Group >
            Value EmitUnary(Environment* env, IROpCode::Enum op, Value const& a, int64_t data) {
                IRef r;
                Trace* trace = getTrace(a);
                trace->liveEnvironments.insert(env);
                if(futureType(a) == Type::Double) {
                    r = trace->EmitUnary(op, Group<Double>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Double>::MA::ValueType), data);
                } else if(futureType(a) == Type::Integer) {
                    r = trace->EmitUnary(op, Group<Integer>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Integer>::MA::ValueType), data);
                } else if(futureType(a) == Type::Logical) {
                    r = trace->EmitUnary(op, Group<Logical>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Logical>::MA::ValueType), data);
                } else _error("Attempting to record invalid type in EmitUnary");
                Value v;
                Future::Init(v, trace, r);
                return v;
            }

        template< template<class X, class Y> class Group >
            Value EmitBinary(Environment* env, IROpCode::Enum op, Value const& a, Value const& b, int64_t data) {
                IRef r;
                Trace* trace = getTrace(a,b);
                trace->liveEnvironments.insert(env);
                if(futureType(a) == Type::Double) {
                    if(futureType(b) == Type::Double)
                        r = trace->EmitBinary(op, Group<Double,Double>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Double,Double>::MA::ValueType), trace->EmitCoerce(trace->GetRef(b), Group<Double,Double>::MB::ValueType), data);
                    else if(futureType(b) == Type::Integer)
                        r = trace->EmitBinary(op, Group<Double,Integer>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Double,Integer>::MA::ValueType), trace->EmitCoerce(trace->GetRef(b), Group<Double,Integer>::MB::ValueType), data);
                    else if(futureType(b) == Type::Logical)
                        r = trace->EmitBinary(op, Group<Double,Logical>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Double,Logical>::MA::ValueType), trace->EmitCoerce(trace->GetRef(b), Group<Double,Logical>::MB::ValueType), data);
                    else _error("Attempting to record invalid type in EmitBinary");
                } else if(futureType(a) == Type::Integer) {
                    if(futureType(b) == Type::Double)
                        r = trace->EmitBinary(op, Group<Integer,Double>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Integer,Double>::MA::ValueType), trace->EmitCoerce(trace->GetRef(b), Group<Integer,Double>::MB::ValueType), data);
                    else if(futureType(b) == Type::Integer)
                        r = trace->EmitBinary(op, Group<Integer,Integer>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Integer,Integer>::MA::ValueType), trace->EmitCoerce(trace->GetRef(b), Group<Integer,Integer>::MB::ValueType), data);
                    else if(futureType(b) == Type::Logical)
                        r = trace->EmitBinary(op, Group<Integer,Logical>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Integer,Logical>::MA::ValueType), trace->EmitCoerce(trace->GetRef(b), Group<Integer,Logical>::MB::ValueType), data);
                    else _error("Attempting to record invalid type in EmitBinary");
                } else if(futureType(a) == Type::Logical) {
                    if(futureType(b) == Type::Double)
                        r = trace->EmitBinary(op, Group<Logical,Double>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Logical,Double>::MA::ValueType), trace->EmitCoerce(trace->GetRef(b), Group<Logical,Double>::MB::ValueType), data);
                    else if(futureType(b) == Type::Integer)
                        r = trace->EmitBinary(op, Group<Logical,Integer>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Logical,Integer>::MA::ValueType), trace->EmitCoerce(trace->GetRef(b), Group<Logical,Integer>::MB::ValueType), data);
                    else if(futureType(b) == Type::Logical)
                        r = trace->EmitBinary(op, Group<Logical,Logical>::R::ValueType, trace->EmitCoerce(trace->GetRef(a), Group<Logical,Logical>::MA::ValueType), trace->EmitCoerce(trace->GetRef(b), Group<Logical,Logical>::MB::ValueType), data);
                    else _error("Attempting to record invalid type in EmitBinary");
                } else _error("Attempting to record invalid type in EmitBinary");
                Value v;
                Future::Init(v, trace, r);
                return v;
            }

        Value EmitSplit(Environment* env, Value const& a, Value const& b, int64_t data) {
            Trace* trace = getTrace(a,b);
            trace->liveEnvironments.insert(env);
            IRef r = trace->EmitSplit(trace->GetRef(a), trace->EmitCoerce(trace->GetRef(b), Type::Integer), data);
            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        Value EmitConstant(Environment* env, Type::Enum type, int64_t length, int64_t c) {
            Trace* trace = getTrace(length);
            trace->liveEnvironments.insert(env);
            IRef r = trace->EmitConstant(type, length, c);
            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        Value EmitRandom(Environment* env, int64_t length) {
            Trace* trace = getTrace(length);
            trace->liveEnvironments.insert(env);
            IRef r = trace->EmitRandom(length);
            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        Value EmitIndex(Environment* env, int64_t length, int64_t a, int64_t b) {
            Trace* trace = getTrace(length);
            trace->liveEnvironments.insert(env);
            IRef r = trace->EmitBinary(IROpCode::add, Type::Integer, trace->EmitIndex(length, a, b), trace->EmitConstant(Type::Integer, length, 1), 0);
            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        Value EmitSequence(Environment* env, int64_t length, int64_t a, int64_t b) {
            Trace* trace = getTrace(length);
            trace->liveEnvironments.insert(env);
            IRef r = trace->EmitSequence(length, a, b);
            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        Value EmitSequence(Environment* env, int64_t length, double a, double b) {
            Trace* trace = getTrace(length);
            trace->liveEnvironments.insert(env);
            IRef r = trace->EmitSequence(length, a, b);
            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        Value EmitGather(Environment* env, Value const& a, Value const& i) {
            Trace* trace = getTrace(i);
            trace->liveEnvironments.insert(env);
            IRef o = trace->EmitConstant(Type::Integer, 1, 1);
            IRef im1 = trace->EmitBinary(IROpCode::sub, Type::Integer, trace->EmitCoerce(trace->GetRef(i), Type::Integer), o, 0);
            IRef r = trace->EmitGather(((Vector const&)a), im1);
            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        Value EmitFilter(Environment* env, Value const& a, Value const& i) {
            Trace* trace = getTrace(a);
            trace->liveEnvironments.insert(env);
            IRef r = trace->EmitFilter(trace->GetRef(a), trace->EmitCoerce(trace->GetRef(i), Type::Logical));
            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        Value EmitIfElse(Environment* env, Value const& a, Value const& b, Value const& cond) {
            Trace* trace = getTrace(a,b,cond);
            trace->liveEnvironments.insert(env);

            IRef r = trace->EmitIfElse( 	trace->GetRef(a),
                    trace->GetRef(b),
                    trace->GetRef(cond));
            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        Value EmitSStore(Environment* env, Value const& a, int64_t index, Value const& b) {
            Trace* trace = getTrace(b);
            trace->liveEnvironments.insert(env);

            IRef m = a.isFuture() ? ((Future const&)a).ref() : trace->EmitSLoad(((Vector const&)a));

            IRef r = trace->EmitSStore(m, index, trace->GetRef(b));

            Value v;
            Future::Init(v, trace, r);
            return v;
        }

        void LiveEnvironment(Environment* env, Value const& a) {
            if(a.isFuture()) {
                Trace* trace = getTrace(a);
                trace->liveEnvironments.insert(env);
            }
        }

        void KillEnvironment(Environment* env) {
            for(std::map<int64_t, Trace*>::const_iterator i = traces.begin(); i != traces.end(); i++) {
                i->second->liveEnvironments.erase(env);
            }
        }

        void Bind(Thread& thread, Value const& v) {
            if(!v.isFuture()) return;
            Trace* trace = ((Future const&)v).trace();
            trace->Execute(thread, ((Future const&)v).ref());
            trace->Reset();
            availableTraces.push_back(trace);
            traces.erase(trace->Size);
        }

        void Flush(Thread & thread) {
            // execute all traces
            for(std::map<int64_t, Trace*>::const_iterator i = traces.begin(); i != traces.end(); i++) {
                Trace* trace = i->second;
                trace->Execute(thread);
                trace->Reset();
                availableTraces.push_back(trace);
            }
            traces.clear();
        }

        void OptBind(Thread& thread, Value const& v) {
            if(!v.isFuture()) return;
            Trace* trace = ((Future const&)v).trace();
            if(trace->nodes.size() > 2048) {
                Bind(thread, v);
            }
        }

        bool isTraceableType(Value const& a) {
            Type::Enum type = futureType(a);
            return type == Type::Double || type == Type::Integer || type == Type::Logical;
        }

        bool isTraceableShape(Value const& a) {
            IRNode::Shape const& shape = futureShape(a);
            return !shape.blocking && shape.length >= TRACE_VECTOR_WIDTH;
        }

        bool isTraceableShape(Value const& a, Value const& b) {
            IRNode::Shape const& shapea = futureShape(a);
            IRNode::Shape const& shapeb = futureShape(b);
            return 	!shapea.blocking &&
                !shapeb.blocking &&
                (shapea.length >= TRACE_VECTOR_WIDTH || shapeb.length >=TRACE_VECTOR_WIDTH) &&
                !(a.isFuture() && b.isFuture() && shapea.length != shapeb.length);
        }

        bool isTraceable(Value const& a) {
            return enabled &&	
                isTraceableType(a) &&
                isTraceableShape(a);
        }

        bool isTraceable(Value const& a, Value const& b) {
            return enabled &&
                isTraceableType(a) && 
                isTraceableType(b) && 
                isTraceableShape(a, b);
        }

        template< template<class X> class Group>
        bool isTraceable(Value const& a) {
            return isTraceable(a);
        }

        template< template<class X, class Y> class Group>
        bool isTraceable(Value const& a, Value const& b) {
            return isTraceable(a, b);
        }

        template< template<class X, class Y, class Z> class Group>
        bool isTraceable(Value const& a, Value const& b, Value const& c) {
            return false;
        }

};

template<>
inline bool Traces::isTraceable<ArithScan>(Value const& a) { return false; }

template<>
inline bool Traces::isTraceable<UnifyScan>(Value const& a) { return false; }

template<>
inline bool Traces::isTraceable<IfElse>(Value const& a, Value const& b, Value const& c) { 
    return enabled &&
        isTraceableType(a) &&
        isTraceableType(b) &&
        isTraceableType(c) &&
        isTraceableShape(a, c) &&
        isTraceableShape(b, c);
}

#endif

