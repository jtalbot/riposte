
#include "../value.h"

#ifdef ENABLE_JIT

#define TRACE_MAX_VECTOR_REGISTERS (32)
#define TRACE_VECTOR_WIDTH (64)
//maximum number of instructions to record before dropping out of the
//recording interpreter
#define TRACE_MAX_RECORDED (1024)

class Thread;

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
		IRef EmitRepeat(int64_t length, int64_t a, int64_t b);
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

#endif

