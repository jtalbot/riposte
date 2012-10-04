# This Makefile requires GNU make.
UNAME := $(shell uname -s)

CXX := clang++ 
IDIRS := -I/opt/local/include -Idependencies
LDIRS := -L/usr/local/lib -L/opt/local/lib

CXXFLAGS := $(IDIRS) -Wall -msse4.1 `llvm-config --cxxflags` -fexceptions
LFLAGS := $(LDIRS) -L. -fpic -lgc -g `llvm-config --ldflags --libs engine bitreader scalaropts`

ifeq ($(UNAME),Linux)
#for clock_gettime
LFLAGS += -lrt
endif

ENABLE_JIT=0
ENABLE_ARBB=0
ENABLE_LIBM=0

SRC := main.cpp type.cpp strings.cpp bc.cpp value.cpp output.cpp interpreter.cpp compiler.cpp internal.cpp parser.cpp coerce.cpp library.cpp runtime.cpp jit.cpp jit_compile.cpp jit_opt.cpp jit_reg.cpp

BC := ops.cpp

EXECUTABLE := bin/riposte

ifeq ($(ENABLE_JIT),1)
	CXXFLAGS += -DENABLE_JIT
	SRC += ir.cpp trace.cpp trace_compile.cpp assembler-x64.cpp
endif

ifneq ($(ENABLE_ARBB),0)
    ARBB_HOME=/opt/intel/arbb/1.0.0.018
    ARBB_EXISTS=$(shell test -d $(ARBB_HOME); echo $$?)

	CXXFLAGS += -I$(ARBB_HOME)/include
	LFLAGS += -L$(ARBB_HOME)/lib/intel64 -larbb
endif

ifneq ($(ENABLE_LIBM),0)
    AMD_LIBM_HOME=/opt/amdlibm-3-0-1-lin64
	
    CXXFLAGS += -I$(AMD_LIBM_HOME)/include -DUSE_AMD_LIBM
	LFLAGS += -L$(AMD_LIBM_HOME)/lib/dynamic -lamdlibm
endif

OBJECTS := $(patsubst %.cpp,bin/%.o,$(SRC))
BYTECODE := $(patsubst %.cpp,bin/%.bc,$(BC)) 
DEPENDENCIES := $(patsubst %.cpp,bin/%.d,$(SRC))

ASM := $(patsubst %.cpp,bin/%.s,$(SRC))

default: debug 

debug: CXXFLAGS += -DDEBUG -O0 -g
debug: ALL

release: CXXFLAGS += -DNDEBUG -O3 -g
release: ALL

irelease: CXXFLAGS += -DNDEBUG -O3 -g
irelease: CXX := icc
irelease: ALL
          
asm: CXXFLAGS += -DNDEBUG -O3 -g
asm: $(ASM)

ALL: $(EXECUTABLE) $(BYTECODE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LFLAGS) -o $@ $^ $(LIBS)

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

bin/%.bc: src/%.cpp
	clang++ $(IDIRS) -O3 -emit-llvm -c $< -o $@

bin/%.s: src/%.cpp
	$(CXX) $(CXXFLAGS) -S -c $< -o $@ 

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) $(DEPENDENCIES) $(BYTECODE)

coverage: CXXFLAGS += -fprofile-arcs -ftest-coverage
coverage: LFLAGS += -fprofile-arcs -ftest-coverage
coverage: debug
	bin/riposte -f tests/coverage.R
	gcov -o bin $(SRC) > /dev/null

coverage_clean:	clean
	rm -f *.gcov bin/*.gcda bin/*.gcno


# dependency rules
bin/%.d:	src/%.cpp
	@$(CXX) $(CXXFLAGS) -MM -MT '$@ $(@:.d=.o)' $< -o $@
	
-include $(DEPENDENCIES)
