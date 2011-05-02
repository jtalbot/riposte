# This Makefile requires GNU make.
UNAME := $(shell uname -s)
 
CXX := g++ 
CXXFLAGS := -Wall -DINSTALL_DIR=$(CURDIR)/bin 
LFLAGS := -L/usr/local/lib -L/opt/local/lib -L. -lm -fpic -lgc

ifeq ($(UNAME),Darwin)
	CXXFLAGS += -I/opt/local/include
endif

SRC := main.cpp type.cpp bc.cpp value.cpp output.cpp interpreter.cpp compiler.cpp internal.cpp parser.cpp

EXECUTABLE := bin/riposte

default: debug

debug: CXXFLAGS += -DDEBUG -O0 -g
debug: $(EXECUTABLE)

release: CXXFLAGS += -DNDEBUG -O3 -g
release: $(EXECUTABLE)

OBJECTS := $(patsubst %.cpp,bin/%.o,$(SRC))
DEPENDENCIES := $(patsubst %.cpp,bin/%.d,$(SRC))

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LFLAGS) -o $@ $^ $(LIBS)

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) $(DEPENDENCIES)

# dependency rules
bin/%.d:	src/%.cpp
	@$(CXX) $(CXXFLAGS) -MM -MT '$@ $(@:.d=.o)' $< -o $@
	
-include $(DEPENDENCIES)
