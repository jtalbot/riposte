CXX = icc
FLAGS = -O3 -msse4.1 -funroll-loops -g -I../../src/Eigen -I/Users/zdevito/Dropbox/vdb-cocoa -vec-report
LFLAGS =

.SUFFIXES:

bin/%:	%.cpp vec.o
	$(CXX) $(LFLAGS) $(FLAGS) $< vec.o -o $@
	
vec.o:	vec.ispc
	ispc --target=sse4 $< -o $@
	
#bin/pca:	../pca.txt
#bin/kmeans:	../kmeans.txt

#../pca.txt:	pca_data.R
#	R -f pca_data.R
#../kmeans.txt:	kmeans_data.R
#	R -f kmeans_data.R
	
clean:
	rm -rf bin/*
