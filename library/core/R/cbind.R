
cbind <- function(deparse.level, ...) {
	l <- list(...)

    if(length(l) == 0L)
        return(NULL)
    
    ncol <- length(l)
	nrow <- max(unlist(lapply(l, length),FALSE,FALSE))
	x <- unlist(lapply(l, function(x) rep_len(x, nrow)),FALSE,FALSE)
	
    dim(x) <- c(nrow, ncol)
    if(!is.null(names(l)) || !is.null(names(l[[1L]])))
        dimnames(x) <- list(rep_len(names(l[[1L]]), nrow), names(l))
	x
}

rbind <- function(deparse.level, ...) {
	l <- list(...)
    
    nrow <- length(l)
    ncol <- max(unlist(lapply(l, length),FALSE,FALSE))
	x <- unlist(lapply(l, function(x) rep_len(x, ncol)),FALSE,FALSE)

    x <- x[nrow*(index(ncol,1L,length(x))-1L)+
                 index(nrow,ncol,length(x))]

	dim(x) <- c(nrow, ncol)
    if(!is.null(names(l)) || !is.null(names(l[[1L]])))
        dimnames(x) <- list(names(l), rep_len(names(l[[1L]]), ncol))
	x
}

