CXX = clang++
FLAGS = -g
LFLAGS =

bin/%:	%.cpp
	$(CXX) $(LFLAGS) $(FLAGS) $< -o $@
	
clean:
	rm -rf bin/*