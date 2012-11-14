# This Makefile requires GNU make.
 
CXXFLAGS := -Wall -msse4.1 -I/opt/local/include
CFLAGS := -Wall -msse4.1
LFLAGS := -L/opt/local/lib -L. -fpic
LIBS := -lgc

ENABLE_EPEE=1

UNAME := $(shell uname -s)
ifeq ($(UNAME),Linux)
#for clock_gettime
LFLAGS += -lrt
endif

SRC := main.cpp type.cpp strings.cpp bc.cpp value.cpp output.cpp interpreter.cpp compiler.cpp internal.cpp parser.cpp coerce.cpp library.cpp

ifeq ($(ENABLE_EPEE),1)
	CXXFLAGS += -DENABLE_EPEE
	SRC += epee/ir.cpp epee/trace.cpp epee/trace_compile.cpp epee/assembler-x64.cpp
endif

EXECUTABLE := riposte
LINENOISE := build/linenoise.o

OBJECTS := $(patsubst %.cpp,build/%.o,$(SRC))
ASM := $(patsubst %.cpp,build/%.s,$(SRC))
DEPENDENCIES := $(patsubst %.cpp,build/%.d,$(SRC))

default: debug

debug: CXXFLAGS += -DDEBUG -O0 -g
debug: $(EXECUTABLE)

release: CXXFLAGS += -DNDEBUG -O3 -g 
release: $(EXECUTABLE)

asm: CXXFLAGS += -DNDEBUG -O3 -g 
asm: $(ASM)

$(EXECUTABLE): $(OBJECTS) $(LINENOISE)
	$(CXX) $(LFLAGS) -o $@ $^ $(LIBS)

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

build/%.s: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -S -c $< -o $@ 

$(LINENOISE): libs/linenoise/linenoise.c
	 $(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) $(LINENOISE) $(DEPENDENCIES)

# dependency rules
build/%.d: src/%.cpp src/%.c
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) -MM -MT '$@ $(@:.d=.o)' $< -o $@
	
-include $(DEPENDENCIES)
