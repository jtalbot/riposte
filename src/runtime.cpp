#include "runtime.h"
#include "coerce.h"

Type::Enum string2Type(String str) {
#define CASE(name, string) if(str == Strings::name) return Type::name;
TYPES(CASE)
#undef CASE
	_error("Invalid type");
}

Double Random(Thread& thread, int64_t const length) {
	//Thread::RandomSeed& r = Thread::seed[thread.index];
	Double o(length);
	for(int64_t i = 0; i < length; i++) {
		/*r.v[0] = r.v[0] * r.m[0] + r.a[0];
		r.v[0] = r.v[0] * r.m[0] + r.a[0];
		r.v[0] = r.v[0] * r.m[0] + r.a[0];

		o[i] = (double)r.v[0] / ((double)std::numeric_limits<uint64_t>::max() + 1);*/

		o[i] = rand() / (double)RAND_MAX;
	}
	return o;
}

template<class D>
void Resize(Thread& thread, bool clone, D& src, int64_t newLength, typename D::Element fill = D::NAelement) {
	if(clone || newLength > src.length) {
		D r(newLength);
		Insert(thread, src, 0, r, 0, std::min(src.length, newLength));
		for(int64_t i = src.length; i < newLength; i++) r[i] = fill;	
		src = r; 
	} else if(newLength < src.length) {
		src.length = newLength;
	} else {
		// No resizing to do, so do nothing...
	}
}

void Resize(Thread& thread, bool clone, Value& src, int64_t newLength) {
	switch(src.type) {
		#define CASE(Name) case Type::Name: { Resize(thread, clone, src, newLength); } break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Resize this type"); break;
	};
}

template<class D>
void Insert(Thread& thread, D const& src, int64_t srcIndex, D& dst, int64_t dstIndex, int64_t length) {
	if((length > 0 && srcIndex+length > src.length) || dstIndex+length > dst.length)
		_error("insert index out of bounds");
	memcpy(dst.v()+dstIndex, src.v()+srcIndex, length*sizeof(typename D::Element));
}

template<class S, class D>
void Insert(Thread& thread, S const& src, int64_t srcIndex, D& dst, int64_t dstIndex, int64_t length) {
	D as = As<D>(thread, src);
	Insert(thread, as, srcIndex, dst, dstIndex, length);
}

template<class D>
void Insert(Thread& thread, Value const& src, int64_t srcIndex, D& dst, int64_t dstIndex, int64_t length) {
	switch(src.type) {
		#define CASE(Name) case Type::Name: Insert(thread, (Name const&)src, srcIndex, dst, dstIndex, length); break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Insert into this type"); break;
	};
}

void Insert(Thread& thread, Value const& src, int64_t srcIndex, Value& dst, int64_t dstIndex, int64_t length) {
	switch(dst.type) {
		#define CASE(Name) case Type::Name: { Insert(thread, src, srcIndex, (Name&) dst, dstIndex, length); } break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Insert into this type"); break;
	};
}

template< class A >
struct SubsetInclude {
	static void eval(Thread& thread, A const& a, Integer const& d, int64_t nonzero, Value& out)
	{
		A r(nonzero);
		int64_t j = 0;
		typename A::Element const* ae = a.v();
		typename Integer::Element const* de = d.v();
		typename A::Element* re = r.v();
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) {
			if(Integer::isNA(de[i])) re[j++] = A::NAelement;	
			else if(de[i] != 0) re[j++] = ae[de[i]-1];
		}
		out = r;
	}
};

template< class A >
struct SubsetExclude {
	static void eval(Thread& thread, A const& a, Integer const& d, int64_t nonzero, Value& out)
	{
		std::set<Integer::Element> index; 
		typename A::Element const* ae = a.v();
		typename Integer::Element const* de = d.v();
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) if(-de[i] > 0 && -de[i] <= (int64_t)a.length) index.insert(-de[i]);
		// iterate through excluded elements copying intervening ranges.
		A r(a.length-index.size());
		typename A::Element* re = r.v();
		int64_t start = 1;
		int64_t k = 0;
		for(std::set<Integer::Element>::const_iterator i = index.begin(); i != index.end(); ++i) {
			int64_t end = *i;
			for(int64_t j = start; j < end; j++) re[k++] = ae[j-1];
			start = end+1;
		}
		for(int64_t j = start; j <= a.length; j++) re[k++] = ae[j-1];
		out = r;
	}
};

template< class A >
struct SubsetLogical {
	static void eval(Thread& thread, A const& a, Logical const& d, Value& out)
	{
		typename A::Element const* ae = a.v();
		typename Logical::Element const* de = d.v();
		// determine length
		int64_t length = 0;
		if(d.length > 0) {
			int64_t j = 0;
			int64_t maxlength = std::max(a.length, d.length);
			for(int64_t i = 0; i < maxlength; i++) {
				if(!Logical::isFalse(de[j])) length++;
				if(++j >= d.length) j = 0;
			}
		}
		A r(length);
		typename A::Element* re = r.v();
		int64_t j = 0, k = 0;
		for(int64_t i = 0; i < std::max(a.length, d.length) && k < length; i++) {
			if(i >= a.length || Logical::isNA(de[j])) re[k++] = A::NAelement;
			else if(Logical::isTrue(de[j])) re[k++] = ae[i];
			if(++j >= d.length) j = 0;
		}
		out = r;
	}
};

template< class A >
inline int64_t find(Thread& thread, A const& a, typename A::Element const& b) {
	typename A::Element const* ae = a.v();
	// eventually use an index here...
	for(int64_t i = 0; i < a.length; i++) {
		if(ae[i] == b) return i;
	}
	return -1;
}

/*template< class A, class B >
struct SubsetIndexed {
	static void eval(Thread& thread, A const& a, B const& b, B const& d, Value& out)
	{
		typename A::Element const* ae = a.v();
		typename B::Element const* de = d.v();
		
		int64_t length = d.length;
		A r(length);
		for(int64_t i = 0; i < length; i++) {
			int64_t index = find(thread, b, de[i]);
			r[i] = index >= 0 ? ae[index] : A::NAelement;
		}
		out = r;
	}
};*/

void SubsetSlow(Thread& thread, Value const& a, Value const& i, Value& out) {
	if(i.isDouble() || i.isInteger()) {
		Integer index = As<Integer>(thread, i);
		int64_t positive = 0, negative = 0;
		for(int64_t i = 0; i < index.length; i++) {
			if(index[i] > 0 || Integer::isNA(index[i])) positive++;
			else if(index[i] < 0) negative++;
		}
		if(positive > 0 && negative > 0)
			_error("mixed subscripts not allowed");
		else if(positive > 0) {
			switch(a.type) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name,...) case Type::Name: SubsetInclude<Name>::eval(thread, (Name const&)a, index, positive, out); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};
		}
		else if(negative > 0) {
			switch(a.type) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetExclude<Name>::eval(thread, (Name const&)a, index, negative, out); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};	
		}
		else {
			switch(a.type) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: out = Name(0); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
			};	
		}
	}
	else if(i.isLogical()) {
		Logical const& index = (Logical const&)i;
		switch(a.type) {
			case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetLogical<Name>::eval(thread, (Name const&)a, index, out); break;
					 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
		};	
	}
	else {
		_error("NYI indexing type");
	}
}

template< class A  >
struct SubsetAssignInclude {
	static void eval(Thread& thread, A const& a, bool clone, Integer const& d, A const& b, Value& out)
	{
		typename A::Element const* be = b.v();
		typename Integer::Element const* de = d.v();

		// compute max index 
		int64_t outlength = a.length;
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) {
			outlength = std::max(outlength, de[i]);
		}

		// should use max index here to extend vector if necessary	
		A r = a;
		Resize(thread, clone, r, outlength);
		typename A::Element* re = r.v();
		for(int64_t i = 0, j = 0; i < length; i++, j++) {	
			if(j >= b.length) j = 0;
			int64_t idx = de[i];
			if(idx != 0)
				re[idx-1] = be[j];
		}
		out = r;
	}
};

template< class A >
struct SubsetAssignLogical {
	static void eval(Thread& thread, A const& a, bool clone, Logical const& d, A const& b, Value& out)
	{
		typename A::Element const* be = b.v();
		typename Logical::Element const* de = d.v();
		
		// determine length
		int64_t length = std::max(a.length, d.length);
		A r = a;
		Resize(thread, clone, r, length);
		typename A::Element* re = r.v();
		int64_t j = 0, k = 0;
		for(int64_t i = 0; i < length; i++) {
			if(i >= a.length && !Logical::isTrue(de[j])) re[i] = A::NAelement;
			else if(Logical::isTrue(de[j])) re[i] = be[k++];
			if(++j >= d.length) j = 0;
			if(k >= b.length) k = 0;
		}
		out = r;
	}
};

/*template< class A, class B >
struct SubsetAssignIndexed {
	static void eval(Thread& thread, A const& a, bool clone, B const& b, B const& d, A const& v, Value& out)
	{
		typename B::Element const* de = d.v();
		
		std::map<typename B::Element, int64_t> overflow;
		int64_t extra = a.length;

		int64_t length = d.length;
		Integer include(length);
		for(int64_t i = 0; i < length; i++) {
			int64_t index = find(thread, b, de[i]);
			if(index < 0) {
				if(overflow.find(de[i]) != overflow.end())
					index = overflow[de[i]];
				else {
					index = extra++;
					overflow[de[i]] = index;
				}
			}
			include[i] = index+1;
		}
		SubsetAssignInclude<A>::eval(thread, a, clone, include, v, out);
	}
};*/

void SubsetAssignSlow(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(i.isDouble() || i.isInteger()) {
		Integer idx = As<Integer>(thread, i);
		switch(a.type) {
#define CASE(Name) case Type::Name: SubsetAssignInclude<Name>::eval(thread, (Name const&)a, clone, idx, As<Name>(thread, b), c); break;
			VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error("NYI: subset assign type"); break;
		};
	}
	else if(i.isLogical()) {
		Logical const& index = (Logical const&)i;
		switch(a.type) {
#define CASE(Name) case Type::Name: SubsetAssignLogical<Name>::eval(thread, (Name const&)a, clone, index, As<Name>(thread, b), c); break;
					 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error(std::string("NYI: Subset of ") + Type::toString(a.type)); break;
		};	
	}
	else {
		_error("NYI indexing type");
	}
}

void Subset2AssignSlow(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(i.length == 1 && (i.isDouble() || i.isInteger())) {
		if(a.isList()) {
			List r = (List&)a;
			int64_t index = asReal1(i)-1;
			if(index >= 0) {
				Resize(thread, clone, r, std::max(index+1, a.length));
				((List&)r)[index] = b;
				c = r;
			}
			else {
				_error("NYI negative indexes on [[");
			}
		} else {
			SubsetAssignSlow(thread, a, clone, i, b, c);
		}
	}
	// Frankly it seems pointless to support this case. We should make this an error.
	//else if(i.isLogical() && i.length == 1 && ((Logical const&)i)[0])
	//	SubsetAssignSlow(thread, a, clone, As<Integer>(thread, i), b, c);
	else
		_error("NYI indexing type");
}


