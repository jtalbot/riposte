CXX = icc
FLAGS = -O3 -fp-model precise -msse4.1 -funroll-loops -g -I../../src/Eigen -I/Users/zdevito/Dropbox/vdb-cocoa -vec-report
LFLAGS =

.SUFFIXES:

bin/%:	%.cpp
	$(CXX) $(LFLAGS) $(FLAGS) $< -o $@
	
#bin/pca:	../pca.txt
#bin/kmeans:	../kmeans.txt

#../pca.txt:	pca_data.R
#	R -f pca_data.R
#../kmeans.txt:	kmeans_data.R
#	R -f kmeans_data.R
	
clean:
	rm -rf bin/*
