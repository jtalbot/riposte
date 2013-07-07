
#include "coerce.h"
#include "value.h"
#include "runtime.h"

#include <dyncall.h>
#include <dlfcn.h>

Type::Enum string2Type(String str) {
#define CASE(name, string, ...) if(str == Strings::name) return Type::name;
TYPES(CASE)
#undef CASE
	_error("Invalid type");
}

Double RandomVector(Thread& thread, int64_t const length) {
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
void Insert(Thread& thread, D const& src, int64_t srcIndex, D& dst, int64_t dstIndex, int64_t length) {
	if((length > 0 && srcIndex+length > src.length()) || dstIndex+length > dst.length())
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
	switch(src.type()) {
		#define CASE(Name) case Type::Name: Insert(thread, (Name const&)src, srcIndex, dst, dstIndex, length); break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Insert into this type"); break;
	};
}

void Insert(Thread& thread, Value const& src, int64_t srcIndex, Value& dst, int64_t dstIndex, int64_t length) {
	switch(dst.type()) {
		#define CASE(Name) case Type::Name: { Insert(thread, src, srcIndex, (Name&) dst, dstIndex, length); } break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Insert into this type"); break;
	};
}

template<class D>
void Resize(Thread& thread, bool clone, D& src, int64_t newLength, typename D::Element fill = D::NAelement) {
	if(clone || newLength > src.length()) {
		D r(newLength);
		Insert(thread, src, 0, r, 0, std::min(src.length(), newLength));
		for(int64_t i = src.length(); i < newLength; i++) r[i] = fill;	
		src = r; 
	} else if(newLength < src.length()) {
		//src.length = newLength;
        _error("NYI: Resize to shorter length");
	} else {
		// No resizing to do, so do nothing...
	}
}

void Resize(Thread& thread, bool clone, Value& src, int64_t newLength) {
	switch(src.type()) {
		#define CASE(Name) case Type::Name: { Resize(thread, clone, src, newLength); } break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Resize this type"); break;
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
		int64_t length = d.length();
		for(int64_t i = 0; i < length; i++) {
			if(Integer::isNA(de[i]) ||
                (de[i]-1) >= a.length()) re[j++] = A::NAelement;
			else if(de[i] != 0) re[j++] = ae[de[i]-1];
		}
		out = r;
	}
};

template<>
struct SubsetInclude<List> {
	static void eval(Thread& thread, List const& a, Integer const& d, int64_t nonzero, Value& out)
	{
		List r(nonzero);
		int64_t j = 0;
		typename List::Element const* ae = a.v();
		typename Integer::Element const* de = d.v();
		typename List::Element* re = r.v();
		int64_t length = d.length();
		for(int64_t i = 0; i < length; i++) {
			if(Integer::isNA(de[i]) ||
                (de[i]-1) >= a.length()) re[j++] = Null::Singleton();
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
		int64_t length = d.length();
		for(int64_t i = 0; i < length; i++) if(-de[i] > 0 && -de[i] <= (int64_t)a.length()) index.insert(-de[i]);
		// iterate through excluded elements copying intervening ranges.
		A r(a.length()-index.size());
		typename A::Element* re = r.v();
		int64_t start = 1;
		int64_t k = 0;
		for(std::set<Integer::Element>::const_iterator i = index.begin(); i != index.end(); ++i) {
			int64_t end = *i;
			for(int64_t j = start; j < end; j++) re[k++] = ae[j-1];
			start = end+1;
		}
		for(int64_t j = start; j <= a.length(); j++) re[k++] = ae[j-1];
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
		if(d.length() > 0) {
			int64_t j = 0;
			int64_t maxlength = std::max(a.length(), d.length());
			for(int64_t i = 0; i < maxlength; i++) {
				if(!Logical::isFalse(de[j])) length++;
				if(++j >= d.length()) j = 0;
			}
		}
		A r(length);
		typename A::Element* re = r.v();
		int64_t j = 0, k = 0;
		for(int64_t i = 0; i < std::max(a.length(), d.length()) && k < length; i++) {
			if(i >= a.length() || Logical::isNA(de[j])) re[k++] = A::NAelement;
			else if(Logical::isTrue(de[j])) re[k++] = ae[i];
			if(++j >= d.length()) j = 0;
		}
		out = r;
	}
};

void SubsetSlow(Thread& thread, Value const& a, Value const& i, Value& out) {
	if(i.isDouble() || i.isInteger()) {
		Integer index = As<Integer>(thread, i);
		int64_t positive = 0, negative = 0;
		for(int64_t i = 0; i < index.length(); i++) {
			if(index[i] > 0 || Integer::isNA(index[i])) positive++;
			else if(index[i] < 0) negative++;
		}
		if(positive > 0 && negative > 0)
			_error("mixed subscripts not allowed");
		else if(positive > 0) {
			switch(a.type()) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name,...) case Type::Name: SubsetInclude<Name>::eval(thread, (Name const&)a, index, positive, out); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type())); break;
			};
		}
		else if(negative > 0) {
			switch(a.type()) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetExclude<Name>::eval(thread, (Name const&)a, index, negative, out); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type())); break;
			};	
		}
		else {
			switch(a.type()) {
				case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: out = Name(0); break;
						 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
				default: _error(std::string("NYI: Subset of ") + Type::toString(a.type())); break;
			};	
		}
	}
	else if(i.isLogical()) {
		Logical const& index = (Logical const&)i;
		switch(a.type()) {
			case Type::Null: out = Null::Singleton(); break;
#define CASE(Name) case Type::Name: SubsetLogical<Name>::eval(thread, (Name const&)a, index, out); break;
					 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error(std::string("NYI: Subset of ") + Type::toString(a.type())); break;
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
		int64_t outlength = a.length();
		int64_t length = d.length();
		for(int64_t i = 0; i < length; i++) {
			outlength = std::max(outlength, de[i]);
		}

		// should use max index here to extend vector if necessary	
		A r = a;
		Resize(thread, clone, r, outlength);
		typename A::Element* re = r.v();
		for(int64_t i = 0, j = 0; i < length; i++, j++) {	
			if(j >= b.length()) j = 0;
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
		int64_t length = std::max(a.length(), d.length());
		A r = a;
		Resize(thread, clone, r, length);
		typename A::Element* re = r.v();
		int64_t j = 0, k = 0;
		for(int64_t i = 0; i < length; i++) {
			if(i >= a.length() && !Logical::isTrue(de[j])) re[i] = A::NAelement;
			else if(Logical::isTrue(de[j])) re[i] = be[k++];
			if(++j >= d.length()) j = 0;
			if(k >= b.length()) k = 0;
		}
		out = r;
	}
};

void SubsetAssignSlow(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(i.isDouble() || i.isInteger()) {
		Integer idx = As<Integer>(thread, i);
		switch(a.type()) {
#define CASE(Name) case Type::Name: SubsetAssignInclude<Name>::eval(thread, (Name const&)a, clone, idx, As<Name>(thread, b), c); break;
			VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error("NYI: subset assign type"); break;
		};
	}
	else if(i.isLogical()) {
		Logical const& index = (Logical const&)i;
		switch(a.type()) {
#define CASE(Name) case Type::Name: SubsetAssignLogical<Name>::eval(thread, (Name const&)a, clone, index, As<Name>(thread, b), c); break;
					 VECTOR_TYPES_NOT_NULL(CASE)
#undef CASE
			default: _error(std::string("NYI: Subset of ") + Type::toString(a.type())); break;
		};	
	}
	else {
		_error("NYI indexing type");
	}
}

void Subset2AssignSlow(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if((i.isDouble() || i.isInteger()) && ((Vector const&)i).length() == 1) {
		if(a.isList()) {
			List r = (List&)a;
			int64_t index = asReal1(i)-1;
			if(index >= 0) {
				Resize(thread, clone, r, std::max(index+1, r.length()));
				r[index] = b;
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

template<class T>
Integer Semijoin(T const& x, T const& table) {
    Integer r(x.length());

    // TODO: turn this into a hash...
    if(table.length() <= x.length()) {
        std::map<typename T::Element, int64_t> index;
        for(int64_t i = table.length()-1; i >= 0; --i) {
            index[table[i]] = i;
        }
        for(int64_t i = 0; i < x.length(); ++i) {
            typename std::map<typename T::Element, int64_t>::const_iterator 
                j = index.find(x[i]);
            r[i] = (j == index.end()) ? Integer::NAelement : (j->second+1);
        }
    }
    else {
        for(int64_t i = 0; i < x.length(); ++i) {
            r[i] = Integer::NAelement;
        }
        std::map< typename T::Element, std::vector<int64_t> > queries;
        for(int64_t i = 0; i < x.length(); ++i) {
            queries[x[i]].push_back(i);
        }
        for(int64_t i = 0; i < table.length() && queries.size() > 0; ++i) {
            typename std::map< typename T::Element, std::vector<int64_t> >::iterator
                j = queries.find(table[i]);
            if(j != queries.end()) {
                std::vector<int64_t> const& q = j->second;
                for(int64_t k = 0; k < q.size(); ++k) {
                    r[q[k]] = (i+1);
                }
                queries.erase(j);
            }
        }
    }

    return r;
}

Integer Semijoin(Value const& a, Value const& b) {

    // assumes that the two arguments are the same type...
    if(a.isNull())
        return Integer(0);
    else if(a.isRaw())
        return Semijoin<Raw>((Raw const&)a, (Raw const&)b);
    else if(a.isLogical())
        return Semijoin<Logical>((Logical const&)a, (Logical const&)b);
    else if(a.isInteger())
        return Semijoin<Integer>((Integer const&)a, (Integer const&)b);
    else if(a.isDouble())
        return Semijoin<Double>((Double const&)a, (Double const&)b);
    else if(a.isCharacter())
        return Semijoin<Character>((Character const&)a, (Character const&)b);
    else
        _error("Unsupported type in semijoin");
}

static void* find_function(Thread& thread, String name) {
    static String lastname = Strings::empty;
    static void* lastfunc = NULL;

    void* func = NULL;
    /*if(std::string(name) == std::string(lastname)) {
        func = lastfunc;
    }
    else {*/
        for(std::map<std::string,void*>::iterator i = thread.state.handles.begin();
            i != thread.state.handles.end(); ++i) {
            func = dlsym(i->second, name);
            if(func != NULL)
                break;
        }
        lastfunc = func;
        lastname = name;
    //}

    if(func == NULL)
        _error("Can't find external function");
    
    return func;
}

template<class T> void arg(DCCallVM* vm, T const& t, int64_t i) 
{ dcArgPointer(vm, (void*) &t); }
template<> void arg<Logical>(DCCallVM* vm, Logical const& t, int64_t i)
{ dcArgChar(vm, t[i % t.length()]); }
template<> void arg<Integer>(DCCallVM* vm, Integer const& t, int64_t i)
{ dcArgLongLong(vm, t[i % t.length()]); }
template<> void arg<Double>(DCCallVM* vm, Double const& t, int64_t i)
{ dcArgDouble(vm, t[i % t.length()]); }
template<> void arg<Character>(DCCallVM* vm, Character const& t, int64_t i)
{ dcArgPointer(vm, (void*)t[i % t.length()]); }
template<> void arg<Raw>(DCCallVM* vm, Raw const& t, int64_t i)
{ dcArgChar(vm, t[i % t.length()]); }
template<> void arg<List>(DCCallVM* vm, List const& t, int64_t i)
{ dcArgPointer(vm, (void*) &(t[i % t.length()])); }

struct Stream {
    virtual ~Stream() {}
    virtual void operator()(DCCallVM* vm, int64_t i) = 0;
};

template<class T>
struct StreamImpl : public Stream {
    T const& v;

    StreamImpl(T const& v) : v(v) {}

    void operator()(DCCallVM* vm, int64_t i) {
        arg(vm, v, i);
    }
};

Stream* MakeStream(Value const& v) {
    if(v.isLogical()) return new StreamImpl<Logical>((Logical const&)v);
    if(v.isInteger()) return new StreamImpl<Integer>((Integer const&)v);
    if(v.isDouble()) return new StreamImpl<Double>((Double const&)v);
    if(v.isCharacter()) return new StreamImpl<Character>((Character const&)v);
    if(v.isRaw()) return new StreamImpl<Raw>((Raw const&)v);
    if(v.isList()) return new StreamImpl<List>((List const&)v);
    return new StreamImpl<Value>(v);
}

struct Unstream {
    virtual ~Unstream() {}
    virtual Value const& value() const = 0;
    virtual void operator()(DCCallVM* vm, int64_t i) = 0;
};

template<class T>
struct UnstreamImpl : public Unstream {
    T v;

    UnstreamImpl(T v) : v(v) {}

    void operator()(DCCallVM* vm, int64_t i) {
        dcArgPointer(vm, &v[i]);
    }
    
    virtual Value const& value() const {
        return v;
    }
};

Unstream* MakeUnstream(String t, int64_t s) {
    if(t == Strings::Logical) return new UnstreamImpl<Logical>(Logical(s));
    if(t == Strings::Integer) return new UnstreamImpl<Integer>(Integer(s));
    if(t == Strings::Double)  return new UnstreamImpl<Double>(Double(s));
    if(t == Strings::Character) return new UnstreamImpl<Character>(Character(s));
    if(t == Strings::Raw)     return new UnstreamImpl<Raw>(Raw(s));
    if(t == Strings::List)    return new UnstreamImpl<List>(List(s));
    _error("Can't unstream a non-vector during a map");
}

List Map(Thread& thread, String func, List args, Character result) {
    // figure out length of result
    int64_t length = 0;
    for(int64_t i = 0; i < args.length(); ++i) {
        if(args[i].isVector())
            length = std::max(length, ((Vector const&)args[i]).length());
        else
            length = std::max(length, (int64_t)1);
    }

    // build up streamers and unstreamers.
    std::vector<Stream*> s;
    for(int64_t i = 0; i < args.length(); ++i) {
        s.push_back(MakeStream(args[i]));
    }
    std::vector<Unstream*> u;
    for(int64_t i = 0; i < result.length(); ++i) {
        u.push_back(MakeUnstream(result[i], length));
    }

    // look up function
    void* f = find_function(thread, func);
    
    // run
    DCCallVM* vm = dcNewCallVM(4096);
    dcMode(vm, DC_CALL_C_DEFAULT);

    for(int64_t i = 0; i < length; ++i) {
        dcReset(vm);
        dcArgPointer(vm, (void*)&thread);
        for(int64_t k = 0; k < s.size(); ++k) {
            (*s[k])(vm, i);
        }
        for(int64_t k = 0; k < u.size(); ++k) {
            (*u[k])(vm, i);
        }
        dcCallVoid(vm, f);
    }
    dcFree(vm);

    List r(result.length());
    for(int64_t i = 0; i < u.size(); ++i) {
        r[i] = u[i]->value();
    }
        
    for(int64_t i = 0; i < s.size(); ++i) {
        delete s[i];
    }
    for(int64_t i = 0; i < u.size(); ++i) {
        delete u[i];
    }
    
    return r;
}
