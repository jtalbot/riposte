#include <stdio.h>
#include <string.h>

// expansion macro for enum value definition
#define ENUM_VALUE(name,string,p) E##name,

#define ENUM_CONST(name, string, EnumType) static const EnumType name;
#define ENUM_CONST_DEFN(name, string, EnumType) const EnumType EnumType::name = {EnumType::E##name};

// expansion macro for enum to string conversion
#define ENUM_CASE(name,string,p) case E##name: return string;

// expansion macro for enum to string conversion
#define ENUM_STRCMP(name,string,p) if (!strcasecmp(str,string)) return name;

/// declare the access function and define enum values
#define DECLARE_ENUM(EnumType,ENUM_DEF) \
struct EnumType { \
  enum Value { \
    ENUM_DEF(ENUM_VALUE,0) \
  }; \
  ENUM_DEF(ENUM_CONST,EnumType) \
  EnumType& operator=(Value const& t) { v = t; return *this; } \
  bool operator==(EnumType const& t) const { return v == t.v; } \
  bool operator!=(EnumType const& t) const { return v != t.v; } \
  char const* toString() const; \
  Value internal() const { return v; } \
  Value v; \
};\

/// define the access function names
#define DEFINE_ENUM(EnumType,ENUM_DEF) \
  ENUM_DEF(ENUM_CONST_DEFN, EnumType) \
  \
  char const* EnumType::toString() const \
  { \
    switch(v) \
    { \
      ENUM_DEF(ENUM_CASE,0) \
      default: return ""; \
    } \
  } \

