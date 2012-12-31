
#ifndef TRACE_H
#define TRACE_H

#ifdef ENABLE_EPEE

#define TRACE_MAX_VECTOR_REGISTERS (32)
#define TRACE_VECTOR_WIDTH (64)
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
		IRef EmitGather(Value const& v, IRef i);
		IRef EmitLoad(Value const& v, int64_t length, int64_t offset);
		IRef EmitSLoad(Value const& v);
		IRef EmitSStore(IRef ref, int64_t index, IRef value);

		IRef GetRef(Value const& v) {
			if(v.isFuture()) return v.future.ref;
			else if(v.length == 1) return EmitConstant(v.type, 1, v.i);
			else return EmitLoad(v,v.length,0);
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

public:

	Type::Enum futureType(Value const& v) const {
		if(v.isFuture()) return v.future.typ;
		else return v.type;
	}

	IRNode::Shape futureShape(Value const& v) const {
		if(v.isFuture()) {
			return traces.find(v.length)->second->nodes[v.future.ref].outShape;
		}
		else 
			return (IRNode::Shape) { v.length, -1, 1, -1 };
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
		return getTrace(a.length);
	}

	Trace* getTrace(Value const& a, Value const& b) {
		int64_t la = a.length;
		int64_t lb = b.length;
		if(la == lb || la == 1)
			return getTrace(lb);
		else if(lb == 1)
			return getTrace(la);
		else
			_error("Shouldn't get here");
	}

	Trace* getTrace(Value const& a, Value const& b, Value const& c) {
		int64_t la = a.length;
		int64_t lb = b.length;
		int64_t lc = c.length;
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
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
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
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}

	Value EmitSplit(Environment* env, Value const& a, Value const& b, int64_t data) {
		Trace* trace = getTrace(a,b);
		trace->liveEnvironments.insert(env);
		IRef r = trace->EmitSplit(trace->GetRef(a), trace->EmitCoerce(trace->GetRef(b), Type::Integer), data);
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}

	Value EmitConstant(Environment* env, Type::Enum type, int64_t length, int64_t c) {
		Trace* trace = getTrace(length);
		trace->liveEnvironments.insert(env);
		IRef r = trace->EmitConstant(type, length, c);
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}

	Value EmitRandom(Environment* env, int64_t length) {
		Trace* trace = getTrace(length);
		trace->liveEnvironments.insert(env);
		IRef r = trace->EmitRandom(length);
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}

	Value EmitIndex(Environment* env, int64_t length, int64_t a, int64_t b) {
		Trace* trace = getTrace(length);
		trace->liveEnvironments.insert(env);
		IRef r = trace->EmitBinary(IROpCode::add, Type::Integer, trace->EmitIndex(length, a, b), trace->EmitConstant(Type::Integer, length, 1), 0);
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}
		
	Value EmitSequence(Environment* env, int64_t length, int64_t a, int64_t b) {
		Trace* trace = getTrace(length);
		trace->liveEnvironments.insert(env);
		IRef r = trace->EmitSequence(length, a, b);
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}

	Value EmitSequence(Environment* env, int64_t length, double a, double b) {
		Trace* trace = getTrace(length);
		trace->liveEnvironments.insert(env);
		IRef r = trace->EmitSequence(length, a, b);
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}

	Value EmitGather(Environment* env, Value const& a, Value const& i) {
		Trace* trace = getTrace(i);
		trace->liveEnvironments.insert(env);
		IRef o = trace->EmitConstant(Type::Integer, 1, 1);
		IRef im1 = trace->EmitBinary(IROpCode::sub, Type::Integer, trace->EmitCoerce(trace->GetRef(i), Type::Integer), o, 0);
		IRef r = trace->EmitGather(a, im1);
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}

	Value EmitFilter(Environment* env, Value const& a, Value const& i) {
		Trace* trace = getTrace(a);
		trace->liveEnvironments.insert(env);
		IRef r = trace->EmitFilter(trace->GetRef(a), trace->EmitCoerce(trace->GetRef(i), Type::Logical));
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}

	Value EmitIfElse(Environment* env, Value const& a, Value const& b, Value const& cond) {
		Trace* trace = getTrace(a,b,cond);
		trace->liveEnvironments.insert(env);
		
		IRef r = trace->EmitIfElse( 	trace->GetRef(a),
						trace->GetRef(b),
						trace->GetRef(cond));
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
		return v;
	}
	
	Value EmitSStore(Environment* env, Value const& a, int64_t index, Value const& b) {
		Trace* trace = getTrace(b);
		trace->liveEnvironments.insert(env);
		
		IRef m = a.isFuture() ? a.future.ref : trace->EmitSLoad(a);

		IRef r = trace->EmitSStore(m, index, trace->GetRef(b));
		
		Value v;
		Future::Init(v, trace->nodes[r].type, trace->nodes[r].shape.length, r);
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
		std::map<int64_t, Trace*>::iterator i = traces.find(v.length);
		if(i == traces.end()) 
			_error("Unevaluated future left behind");
		Trace* trace = i->second;
		trace->Execute(thread, v.future.ref);
		trace->Reset();
		availableTraces.push_back(trace);
		traces.erase(i);
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
		std::map<int64_t, Trace*>::iterator i = traces.find(v.length);
		if(i == traces.end()) 
			_error("Unevaluated future left behind");
		Trace* trace = i->second;
		if(trace->nodes.size() > 2048) {
			Bind(thread, v);
		}
	}

};

#endif

#endif
