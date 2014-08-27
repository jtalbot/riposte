
#include <algorithm>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

template<class T, bool nalast, bool descending>
struct compare {
    compare(T const& t) : t(t) {}
    T const& t;

    bool operator()(Integer::Element ai, Integer::Element bi) const {
        typename T::Element const& a = t[ai];
        typename T::Element const& b = t[bi];

        if(!T::isNA(a) && !T::isNA(b))
            return descending ? (a > b) : (a < b);
        else if(T::isNA(a) && !T::isNA(b))
            return !nalast;
        else if(!T::isNA(a) && T::isNA(b))
            return nalast;
        else
            return false;
    }
};

template<bool nalast, bool descending>
struct compare<Character, nalast, descending> {
    compare(Character const& t) : t(t) {}
    Character const& t;

    bool operator()(Integer::Element ai, Integer::Element bi) const {
        typename Character::Element const& a = t[ai];
        typename Character::Element const& b = t[bi];

        if(!Character::isNA(a) && !Character::isNA(b))
            return descending ? (strcmp(a->s,b->s)>0) : (strcmp(a->s,b->s)<0);
        else if(Character::isNA(a) && !Character::isNA(b))
            return !nalast;
        else if(!Character::isNA(a) && Character::isNA(b))
            return nalast;
        else
            return false;
    }
};

template<bool nalast, bool descending>
static Integer order(Vector const& v) {
    Integer result(v.length());
    for(int64_t i = 0; i < v.length(); ++i)
        result[i] = i;
    
    switch(v.type()) {
        case Type::Raw:
            std::stable_sort(&result[0], &result[v.length()],
                compare<Raw, nalast, descending>((Raw const&)v));
            break;
        case Type::Logical:
            std::stable_sort(&result[0], &result[v.length()],
                compare<Logical, nalast, descending>((Logical const&)v));
            break;
        case Type::Integer:
            std::stable_sort(&result[0], &result[v.length()],
                compare<Integer, nalast, descending>((Integer const&)v));
            break;
        case Type::Double:
            std::stable_sort(&result[0], &result[v.length()],
                compare<Double, nalast, descending>((Double const&)v));
            break;
        case Type::Character:
            std::stable_sort(&result[0], &result[v.length()],
                compare<Character, nalast, descending>((Character const&)v));
            break;
        default:
            _error("unimplemented type 'list' in 'order'");
    }

    for(int64_t i = 0; i < v.length(); ++i)
        result[i]++;

    return result;    
}

struct compare2 {
    compare2 const* next;
    compare2(compare2 const* next=0) : next(next) {}
    virtual ~compare2() { if(next) delete next; }
    virtual bool operator()(Integer::Element ai, Integer::Element bi) const = 0;
};

template<class T, bool nalast, bool descending>
struct compare2_impl : public compare2 {
    T const& t;
    compare2_impl(compare2 const* next, T const& t) : compare2(next), t(t) {}

    virtual bool operator()(Integer::Element ai, Integer::Element bi) const {
        typename T::Element const& a = t[ai];
        typename T::Element const& b = t[bi];

        if(!T::isNA(a) && !T::isNA(b)) {
            if(a == b && next) return next->operator()(ai, bi);
            else return descending ? (a > b) : (a < b);
        }
        else if(T::isNA(a) && !T::isNA(b))
            return !nalast;
        else if(!T::isNA(a) && T::isNA(b))
            return nalast;
        else
            return false;
    }
};

template<bool nalast, bool descending>
struct compare2_impl<Character, nalast, descending> : public compare2 {
    Character const& t;
    compare2_impl(compare2 const* next, Character const& t) : compare2(next), t(t) {}

    virtual bool operator()(Integer::Element ai, Integer::Element bi) const {
        typename Character::Element const& a = t[ai];
        typename Character::Element const& b = t[bi];

        if(!Character::isNA(a) && !Character::isNA(b)) {
            if(a == b && next) return next->operator()(ai, bi);
            else return descending ? (strcmp(a->s,b->s)>0) : (strcmp(a->s,b->s)<0);
        }
        else if(Character::isNA(a) && !Character::isNA(b))
            return !nalast;
        else if(!Character::isNA(a) && Character::isNA(b))
            return nalast;
        else
            return false;
    }
};

struct compare_chain {
    compare_chain() : c(0) {}
    compare2 const* c;

    bool operator()(Integer::Element ai, Integer::Element bi) const {
        if(c) return c->operator()(ai, bi);
        else return false;
    }

    template<bool nalast, bool descending>
    void push(Vector const& v) {
    switch(v.type()) {
        case Type::Raw:
            c = new compare2_impl<Raw, nalast, descending>(c, (Raw const&)v);
            break;
        case Type::Logical:
            c = new compare2_impl<Logical, nalast, descending>(c, (Logical const&)v);
            break;
        case Type::Integer:
            c = new compare2_impl<Integer, nalast, descending>(c, (Integer const&)v);
            break;
        case Type::Double:
            c = new compare2_impl<Double, nalast, descending>(c, (Double const&)v);
            break;
        case Type::Character:
            c = new compare2_impl<Character, nalast, descending>(c, (Character const&)v);
            break;
        default:
            _error("unimplemented type 'list' in 'order'");
    }
}

};

extern "C"
Value order(State& state, Value const* args) {
    bool nalast = Logical::isTrue(((Logical const&)args[0])[0]);
    bool descending = Logical::isTrue(((Logical const&)args[1])[0]);
    List const& l = (List const&)args[2];

    if(l.length() == 0)
        return Integer(0);  // this differs from R, but seems nicer.

    // If only 1, do the fast version with no virtual function calls
    if(l.length() == 1 && l[0].isVector()) {
        if(nalast) {
            if(descending)
                return order<true, true>((Vector const&)l[0]);
            else 
                return order<true, false>((Vector const&)l[0]);
        }
        else { 
            if(descending)
                return order<false, true>((Vector const&)l[0]);
            else 
                return order<false, false>((Vector const&)l[0]);
        }
    }

    // TODO: even in this case we should be able to avoid
    // virtual function calls for the first sort vector,
    // which would probably greatly speed things up.
    compare_chain chain;
    int64_t length;
    for(int64_t i = l.length()-1; i >= 0; --i) {
        length = ((Vector const&)l[i]).length();
        if(nalast) {
            if(descending)
                chain.push<true, true>((Vector const&)l[i]);
            else 
                chain.push<true, false>((Vector const&)l[i]);
        }
        else { 
            if(descending)
                chain.push<false, true>((Vector const&)l[i]);
            else 
                chain.push<false, false>((Vector const&)l[i]);
        }
    }
    
    Integer result(length);
    for(int64_t i = 0; i < length; ++i)
        result[i] = i;
            
    std::stable_sort(&result[0], &result[length], chain);

    delete chain.c;

    for(int64_t i = 0; i < length; ++i)
        result[i]++;

    return result;    
}

