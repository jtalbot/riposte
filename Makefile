# This Makefile requires GNU make.
UNAME := $(shell uname -s)
 
CXX := g++
CXXFLAGS := -Wall -DRIPOSTE_DISABLE_TRACING
LFLAGS := -L/usr/local/lib -L/opt/local/lib -L. -fpic -lgc -g

ENABLE_ARBB=0
ARBB_HOME=/opt/intel/arbb/1.0.0.018
ARBB_EXISTS=$(shell test -d $(ARBB_HOME); echo $$?)

ifeq ($(ENABLE_ARBB),0)
	CXXFLAGS += -I/opt/local/include
else
	LFLAGS += -L$(ARBB_HOME)/lib/intel64 -larbb -ltbb
	CXXFLAGS += -I$(ARBB_HOME)/include
endif

SRC := main.cpp type.cpp bc.cpp value.cpp output.cpp interpreter.cpp compiler.cpp internal.cpp parser.cpp coerce.cpp library.cpp ir.cpp recording.cpp trace.cpp

EXECUTABLE := bin/riposte

OBJECTS := $(patsubst %.cpp,bin/%.o,$(SRC))
DEPENDENCIES := $(patsubst %.cpp,bin/%.d,$(SRC))

ASM := $(patsubst %.cpp,bin/%.s,$(SRC))

default: debug

debug: CXXFLAGS += -DDEBUG -O0 -g
debug: $(EXECUTABLE)

release: CXXFLAGS += -DNDEBUG -O3 -g -ftree-vectorize 
release: $(EXECUTABLE)

irelease: CXXFLAGS += -DNDEBUG -O3 -g -ftree-vectorize
irelease: CXX := icc
irelease: $(EXECUTABLE)
          
asm: CXXFLAGS += -DNDEBUG -O3 -g -ftree-vectorize 
asm: $(ASM)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LFLAGS) -o $@ $^ $(LIBS)

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

bin/%.s: src/%.cpp
	$(CXX) $(CXXFLAGS) -S -c $< -o $@ 

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) $(DEPENDENCIES)

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
