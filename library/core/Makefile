# This Makefile requires GNU make.
 
UNAME := $(shell uname -s)
CXX := clang++
CXXFLAGS := -Wall -I/opt/local/include
CFLAGS := -Wall -fPIC
LFLAGS := -dynamiclib -L/opt/local/lib
LIBS := -lpthread -ltre

SRC := core.cpp print.cpp primitive.cpp connections.cpp datetime.cpp fileinfo.cpp getwd.cpp grep.cpp sprintf.cpp substr.cpp

EXECUTABLE := libs/core.dylib

OBJECTS := $(patsubst %.cpp,build/%.o,$(SRC))
DEPENDENCIES := $(patsubst %.cpp,build/%.d,$(SRC))

default: debug

debug: CXXFLAGS += -DDEBUG -O0 -g
debug: $(EXECUTABLE)

release: CXXFLAGS += -DNDEBUG -O4 -g
release: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LFLAGS) -o $@ $^ ../../riposte.dylib $(LIBS)

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

.PHONY: clean
clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) $(DEPENDENCIES)

# dependency rules
build/%.d: src/%.cpp
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) -MM -MT '$@ $(@:.d=.o)' $< -o $@
	
-include $(DEPENDENCIES)

