# This Makefile requires GNU make.
 
UNAME := $(shell uname -s)
CXX := clang++
CC := clang
CXXFLAGS := -std=c++11 -Wall -Iinclude -g
CFLAGS := -Wall -Iinclude -g
LFLAGS := -fpic
LIBS := -Llibs/dyncall/dyncall -lpthread -ldyncall_s

ifeq ($(UNAME),Linux)
	LIBS += -lrt
endif

EPEE=0

SRC := type.cpp strings.cpp bc.cpp value.cpp output.cpp interpreter.cpp compiler.cpp runtime.cpp library.cpp format.cpp gc.cpp call.cpp thread.cpp inst.cpp

SRC += parser/lexer.cpp

ifeq ($(EPEE),1)
	CXXFLAGS += -DEPEE
	SRC += epee/ir.cpp epee/trace.cpp epee/trace_compile.cpp epee/assembler-x64.cpp
endif

API_SRC := api/api.cpp api/Applic.cpp api/Connections.cpp api/Defn.cpp api/Error.cpp api/Fileio.cpp api/Linpack.cpp api/Print.cpp api/R.cpp api/Rinterface.cpp api/Rinternals.cpp api/Arith.cpp api/eventloop.cpp api/GraphicsDevice.cpp api/GraphicsEngine.cpp api/Memory.cpp api/PrtUtil.cpp api/Random.cpp api/Rdynload.cpp api/Riconv.cpp api/Rmath.cpp api/Utils.cpp api/RS.cpp


EXECUTABLE := riposte
RIPOSTE := libRiposte.dylib
API := libR.dylib
MAIN := build/main.o
LINENOISE := build/linenoise.o
DYNCALL := libs/dyncall/dyncall/libdyncall_s.a
PACKAGES:= core

ALL_SRC := $(SRC)
ALL_SRC += $(API_SRC)
ALL_SRC += main.cpp

OBJECTS := $(patsubst %.cpp,build/%.o,$(SRC))
ASM := $(patsubst %.cpp,build/%.s,$(SRC))

API_OBJECTS := $(patsubst %.cpp,build/%.o,$(API_SRC))

DEPENDENCIES := $(patsubst %.cpp,build/%.d,$(ALL_SRC))

default: debug

debug: CXXFLAGS += -DDEBUG -O0
debug: all

release: CXXFLAGS += -DNDEBUG -O3
release: all

asm: CXXFLAGS += -DNDEBUG -O3
asm: $(ASM)

all: $(EXECUTABLE) $(API) $(PACKAGES)

$(EXECUTABLE): $(MAIN) $(LINENOISE) $(DYNCALL) $(RIPOSTE)
	$(CXX) $(LFLAGS) -L. -o $@ $^ $(LIBS)

$(PACKAGES): $(RIPOSTE)
	$(MAKE) -C library/$@ $(MAKECMDGOALS)

$(API): $(API_OBJECTS) $(RIPOSTE)
	$(CXX) $(LFLAGS) -L/usr/local/opt/gettext/lib/ -L. -lRiposte -Xlinker -reexport-llzma -Xlinker -reexport-lintl -Xlinker -reexport-lRmath -dynamiclib -compatibility_version 3.2.0 -current_version 3.2.1 -o $@ $^

$(RIPOSTE): $(OBJECTS)
	$(CXX) $(LFLAGS) -dynamiclib -o $@ $^ $(LIBS)

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

build/%.s: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -S -c $< -o $@ 

$(LINENOISE): libs/linenoise/linenoise.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(DYNCALL):
	$(MAKE) -C libs/dyncall -f Makefile.embedded

.PHONY: clean
clean:
	rm -rf $(EXECUTABLE) $(RIPOSTE) $(API) $(MAIN) $(OBJECTS) $(API_OBJECTS) $(LINENOISE) $(DEPENDENCIES)
	$(MAKE) -C library/core $(MAKECMDGOALS)
	$(MAKE) -C libs/dyncall -f Makefile.embedded clean

# dependency rules
build/%.d: src/%.cpp
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) -MM -MT '$@ $(@:.d=.o)' $< -o $@
	
-include $(DEPENDENCIES)


# tests
COVERAGE_TESTS = $(shell find tests/coverage -type f -name '*.R')
BLACKBOX_TESTS = $(shell find tests/blackbox -type f -name '*.R')
CORE_TESTS = $(shell find library/core/tests -type f -name '*.R')
#BASE_TESTS = $(shell find library/base/tests -type f -name '*.R')

.PHONY: tests $(COVERAGE_TESTS) $(BLACKBOX_TESTS) $(CORE_TESTS) #$(BASE_TESTS)
COVERAGE_FLAGS := 
tests: COVERAGE_FLAGS += >/dev/null
tests: $(COVERAGE_TESTS) $(BLACKBOX_TESTS) $(CORE_TESTS) #$(BASE_TESTS)

$(COVERAGE_TESTS):
	-@Rscript --vanilla --default-packages=NULL $@ > $@.key
	-@./riposte --format=R -f $@ > $@.out
	-@diff -b $@.key $@.out $(COVERAGE_FLAGS)
	-@rm $@.key $@.out

$(BLACKBOX_TESTS):
	-@Rscript --vanilla --default-packages=NULL $@ > $@.key
	-@./riposte --format=R -f $@ > $@.out
	-@diff -b $@.key $@.out $(COVERAGE_FLAGS)
	-@rm $@.key $@.out

$(CORE_TESTS):
	-@Rscript --vanilla --default-packages=NULL $@ > $@.key
	-@./riposte --format=R -f $@ > $@.out
	-@diff -b $@.key $@.out $(COVERAGE_FLAGS)
	-@rm $@.key $@.out

$(BASE_TESTS):
	-@Rscript --vanilla --default-packages=NULL $@ > $@.key
	-@./riposte --format=R -f $@ > $@.out
	-@diff -b $@.key $@.out $(COVERAGE_FLAGS)
	-@rm $@.key $@.out

