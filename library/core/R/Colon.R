
`:` <- function(from, to) { 
	if(to > from) (seq_len(to-from+1L)-1L)+from
	else if(to < from) (1L-seq_len(from-to+1L))+from
	else from
}

