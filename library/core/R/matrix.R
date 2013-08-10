
matrix <- function(data, nrow, ncol, byrow, dimnames, missing.nrow, missing.ncol) {
	if(missing.nrow && missing.ncol) {
        ncol <- 1L
        nrow <- length(data)
    }
    else if(missing.nrow) {
        nrow <- ceiling(length(data)/ncol)
    }
    else if(missing.ncol) {
        ncol <- ceiling(length(data)/nrow)
    }
    nrow <- as.integer(nrow)
    ncol <- as.integer(ncol)

    if((length(data) != 0) && ((nrow*ncol) %% length(data) != 0L))
        warning(TRUE,FALSE,sprintf('data length [%d] is not a sub-multiple or multiple of the number of rows [%d]', length(data), nrow))

    if(length(data) < nrow*ncol)
		data <- rep_len(data, nrow*ncol)

    if(length(r) > nrow*ncol)
        r <- r[seq_len(nrow*ncol)]

    if(byrow)
		data <- data[nrow*(index(ncol,1L,length(data))-1L)+
                        index(nrow,ncol,length(data))]

    dim(data) <- c(nrow, ncol)
    dimnames(data) <- dimnames
    data    
}

is.matrix <- function(x)
    length(attr(x, 'dim')) == 2L

`+.matrix` <- function(x,y) {
	xd <- dim(x)
	yd <- dim(y)
	if(!all(xd == yd))
		.stop("Matrices are not conformable")
	r <- strip(x) + strip(y)
	matrix(r, xd[[1L]], xd[[2L]])
}

`*.matrix` <- function(x,y) {
	xd <- dim(x)
	yd <- dim(y)
	if(!all(xd == yd))
		.stop("Matrices are not conformable")
	r <- strip(x) * strip(y)
	matrix(r, xd[[1L]], xd[[2L]])
}

`/.matrix` <- function(x,y) {
	xd <- dim(x)
	yd <- dim(y)
	if(!all(xd == yd))
		.stop("Matrices are not conformable")
	r <- strip(x) / strip(y)
	if(is.null(yd))
		matrix(r, xd[[1L]], xd[[2L]])
	else
		matrix(r, yd[[1L]], yd[[2L]])
}

`<=.matrix` <- function(x,y) {
	xd <- dim(x)
	yd <- dim(y)
	if(!all(xd == yd))
		.stop("Matrices are not conformable")
	r <- strip(x) <= strip(y)
	matrix(r, xd[[1L]], xd[[2L]])
}

`<.matrix` <- function(x,y) {
	xd <- dim(x)
	yd <- dim(y)
	if(!all(xd == yd))
		.stop("Matrices are not conformable")
	r <- strip(x) < strip(y)
	matrix(r, xd[[1L]], xd[[2L]])
}

`sqrt.matrix` <- function(x) {
	xd <- dim(x)
	matrix(sqrt(strip(x)), xd[[1L]], xd[[2L]])
}

