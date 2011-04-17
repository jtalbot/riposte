# This Makefile requires GNU make.
UNAME := $(shell uname -s)
 
CXX := g++ 
CXXFLAGS := -Wall -DINSTALL_DIR=$(CURDIR)/bin `R CMD config --cppflags`
LFLAGS := -L/usr/local/lib -L/opt/local/lib -L. -lm -fpic -lgc `R CMD config --ldflags`

ifeq ($(UNAME),Darwin)
	CXXFLAGS += -I/opt/local/include
endif

SRC := main.cpp extras.cpp type.cpp bc.cpp parse.cpp output.cpp interpreter.cpp compiler.cpp internal.cpp parser.cpp

EXECUTABLE := bin/riposte

default: release

debug: CXXFLAGS += -DDEBUG -O0 -g
debug: $(EXECUTABLE)

release: CXXFLAGS += -DNDEBUG -O3 -g
release: $(EXECUTABLE)

OBJECTS := $(patsubst %.cpp,bin/%.o,$(SRC))

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LFLAGS) -o $@ $^ $(LIBS)

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS)
