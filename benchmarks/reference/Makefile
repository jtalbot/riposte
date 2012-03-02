CXX = clang++
FLAGS = -O3 -funroll-loops -g -I../../src/Eigen
LFLAGS =

.SUFFIXES:

bin/%:	%.cpp
	$(CXX) $(LFLAGS) $(FLAGS) $< -o $@
	
bin/pca:	../pca.txt

../pca.txt:	pca_data.R
	R -f pca_data.R
	
clean:
	rm -rf bin/*