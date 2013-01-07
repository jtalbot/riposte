#include <stdio.h>

// the ... arguments to these macros allow enum definitions to support enum macros that contain additional arguments

// expansion macro for enum value definition
#define ENUM_ENTRY(name,string,...) name,

/// declare the access function and define enum values
#define DECLARE_ENUM(EnumType,ENUM_DEF) \
namespace EnumType { \
  enum Enum { \
    ENUM_DEF(ENUM_ENTRY) \
  }; \
  char const* toString(Enum e); \
  Enum toEnum(char const* str); \
}

#define ENUM_ENTRY_WITH_VALUE(name,string,value,...) name=value,

#define DECLARE_ENUM_WITH_VALUES(EnumType,ENUM_DEF) \
namespace EnumType { \
  enum Enum { \
    ENUM_DEF(ENUM_ENTRY_WITH_VALUE) \
  }; \
  char const* toString(Enum e); \
  Enum toEnum(char const* str); \
}


// expansion macro for enum to string conversion
#define ENUM_TO_STRING(name,string,...) case name: return string;

#define DEFINE_ENUM_TO_STRING(EnumType,ENUM_DEF) \
namespace EnumType { \
  char const* toString(Enum e) \
  { \
    switch(e) \
    { \
      ENUM_DEF(ENUM_TO_STRING) \
      default: throw "Attempt to cast invalid type to string"; \
    } \
  } \
}

// expansion macro for string to enum conversion
#define STRING_TO_ENUM(name,string,...) if (!strcasecmp(str,string)) return name;

#define DEFINE_STRING_TO_ENUM(EnumType,ENUM_DEF) \
namespace EnumType { \
  Enum toEnum(char const* str) \
  { \
    ENUM_DEF(STRING_TO_ENUM) \
    throw "Invalid cast of string to " # EnumType; \
  } \
}

