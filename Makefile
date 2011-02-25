# This Makefile requires GNU make.
CC := g++ 
CFLAGS := -g -O3 -Wall -DNDEBUG -I/usr/local/include -L/usr/local/lib -L/opt/local/lib -L. -DINSTALL_DIR=$(CURDIR)/bin `R CMD config --cppflags`
#CFLAGS := -g -O0 -I/usr/local/include -L/usr/local/lib -L/opt/local/lib -L. -DINSTALL_DIR=$(CURDIR)/bin `R CMD config --cppflags`
LFLAGS := -lm -fpic -lRmath -lgc `R CMD config --ldflags`
ROCKY_OBJS := src/main.o src/extras.o src/type.o src/output.o src/parse.o src/interpreter.o src/compiler.o src/internal.o src/bc.o

PROGS := bin/riposte

.PHONY: default all clean

default: all

all: $(PROGS)

bin/riposte: src/main.o src/extras.o src/type.o src/output.o src/parse.o src/interpreter.o src/compiler.o src/internal.o src/bc.o
	$(CC) $(CFLAGS) $(LFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ 

clean:
	rm -rf $(PROGS) $(ROCKY_OBJS)
