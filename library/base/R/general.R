
force <- function(x) x

mapply <- function(FUN, ...) {
	lapply(t.list(...), FUN)
}

paste <- function(..., sep = " ", collapse = NULL) {
	r <- mapply(function(x) .Internal(paste)(x, sep), ...)
	if(!is.null(collapse)) .Internal(paste)(r, collapse)
	else r
}

anyDuplicated <- function(x) {
	for(i in seq_len(length(x)-1)) {
		for(j in (i+1):length(x)) {
			if(x[[i]] == x[[j]]) return(j)
		}
	}
	0 
}

c <- function(...) unlist(list(...))

make.names <- function(x) {
	x
}
