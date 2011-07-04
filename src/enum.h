#include <stdio.h>
#include <string.h>

// the ... arguments to these macros allow enum definitions to support enum macros that contain additional arguments
// expansion macro for enum value definition
#define ENUM_VALUE(name,string,p,...) E_##name,

#define ENUM_CONST(name, string, EnumType,...) static const EnumType name;
#define ENUM_CONST_DEFN(name, string, EnumType,...) const EnumType EnumType::name = {EnumType::E_##name};

// expansion macro for enum to string conversion
#define ENUM_CASE(name,string,p,...) case E_##name: return string;

// expansion macro for enum to string conversion
#define ENUM_STRCMP(name,string,p,...) if (!strcasecmp(str,string)) return name;

/// declare the access function and define enum values
#define DECLARE_ENUM(EnumType,ENUM_DEF) \
struct EnumType { \
  enum EnumValue { \
    ENUM_DEF(ENUM_VALUE,0) \
  }; \
  ENUM_DEF(ENUM_CONST,EnumType) \
  EnumType& operator=(EnumValue const& t) { v = t; return *this; } \
  bool operator==(EnumType const& t) const { return v == t.v; } \
  bool operator!=(EnumType const& t) const { return v != t.v; } \
  char const* toString() const; \
  EnumValue Enum() const { return v; } \
  EnumValue v; \
};\

/// define the access function names
#define DEFINE_ENUM(EnumType,ENUM_DEF) \
  ENUM_DEF(ENUM_CONST_DEFN, EnumType) \

#define DEFINE_ENUM_TO_STRING(EnumType,ENUM_DEF) \
  char const* EnumType::toString() const \
  { \
    switch(v) \
    { \
      ENUM_DEF(ENUM_CASE,0) \
      default: return ""; \
    } \
  } \

