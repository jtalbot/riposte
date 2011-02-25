#include <stdio.h>
#include <string.h>

// expansion macro for enum value definition
#define ENUM_VALUE(name,string) name,

// expansion macro for enum to string conversion
#define ENUM_CASE(name,string) case name: return string;

// expansion macro for enum to string conversion
#define ENUM_STRCMP(name,string) if (!strcasecmp(str,string)) return name;

/// declare the access function and define enum values
#define DECLARE_ENUM(EnumType,ENUM_DEF) \
class EnumType { \
  public: \
  enum Value { \
    ENUM_DEF(ENUM_VALUE) \
  }; \
  EnumType() {v = (Value)0;} \
  EnumType(Value const& t) {v = t;} \
  bool operator==(Value const& t) const { return v == t; } \
  bool operator==(EnumType const& t) const { return v == t.v; } \
  bool operator!=(Value const& t) const { return v != t; } \
  bool operator!=(EnumType const& t) const { return v != t.v; } \
  char const* toString() const; \
  Value internal() const { return v; } \
  private: \
    Value v; \
  };\

/// define the access function names
#define DEFINE_ENUM(EnumType,ENUM_DEF) \
  char const* EnumType::toString() const \
  { \
    switch(v) \
    { \
      ENUM_DEF(ENUM_CASE) \
      default: return ""; \
    } \
  } \

