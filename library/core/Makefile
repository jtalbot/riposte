# This Makefile requires GNU make.
 
UNAME := $(shell uname -s)
CXX := clang++
CXXFLAGS := -std=c++11 -Wall -Wno-undefined-bool-conversion -I/usr/local/include -I../../include -g
CFLAGS := -Wall -fPIC -g
LFLAGS := -dynamiclib -L/usr/local/lib -L../.. -L../../libs/dyncall/dyncall
LIBS := -lpthread -ltre -lpcre -lboost_iostreams-mt -llzma -liconv -ldyncall_s

SRC := core.cpp print.cpp primitive.cpp connections.cpp datetime.cpp fileinfo.cpp getwd.cpp grep.cpp sprintf.cpp substr.cpp getenv.cpp setenv.cpp files.cpp basename.cpp locales.cpp parse.cpp chartr.cpp order.cpp solve.cpp mapply.cpp tabulate.cpp rawConversion.cpp base-internal.cpp ns-internals.cpp dynload.cpp callexternal.cpp iconv.cpp listfiles.cpp Systime.cpp unlink.cpp Special.cpp system.cpp

EXECUTABLE := libs/core.dylib

OBJECTS := $(patsubst %.cpp,build/%.o,$(SRC))
DEPENDENCIES := $(patsubst %.cpp,build/%.d,$(SRC))

default: debug

debug: CXXFLAGS += -DDEBUG -O0
debug: $(EXECUTABLE)

release: CXXFLAGS += -DNDEBUG -O3
release: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LFLAGS) -o $@ $^ ../../libRiposte.dylib ../../libR.dylib $(LIBS)

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

