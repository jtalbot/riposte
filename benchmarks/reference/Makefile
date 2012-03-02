CXX = clang++
FLAGS = -g
LFLAGS =

.SUFFIXES:

bin/%:	%.cpp
	$(CXX) $(LFLAGS) $(FLAGS) $< -o $@
	
clean:
	rm -rf bin/*