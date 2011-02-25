
struct Type {
	enum Value {X,Y,Z};
	Value value;
	Type(Value const& v) {value=v;}
	bool operator==(Value const& v) {return value == v;}
	bool operator==(Type const& t) {return value == t.value;}
};

int main(void) {
	Type a = Type::X;
	Type b = Type::Y;

	if(a == b) {
		return 1;
	}
	return 0;
}
