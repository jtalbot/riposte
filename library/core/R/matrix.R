
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

    if((nrow*ncol) %% length(data) != 0L)
        warning(TRUE,FALSE,sprintf('data length [%d] is not a sub-multiple or multiple of the number of rows [%d]', length(data), nrow))

    if(length(data) < nrow*ncol)
		data <- rep_len(data, nrow*ncol)

    if(byrow)
		data <- data[nrow*(index(ncol,1L,length(data))-1L)+
                        index(nrow,ncol,length(data))]

    dim(data) <- c(nrow, ncol)
    dimnames(data) <- dimnames
    data    
}

is.matrix <- function(x)
    length(attr(x, 'dim')) == 2L

`%*%` <- function(x,y) {

	xd <- dim(x)
	if(is.null(xd)) xd <- c(1L, length(x))
	yd <- dim(y)
	if(is.null(yd)) yd <- c(length(y), 1L)

	if(xd[[2L]] != yd[[1L]])
		.stop("Matrices are not conformable")

	if(xd[[1L]] == 1L && yd[[2L]] == 1L) {
		return(sum(strip(x)*strip(y)))
	} else if(xd[[1L]] == 1L) {
		r <- double(0)
		for(i in 1L:yd[[2L]])
			r[[i]] <- sum(strip(x)*strip(y[,i]))
		return(r)
	}
	else if(yd[[2L]] == 1L) {
		r <- 0
		for(i in 1L:xd[[2L]]) {
			r <- r + strip(x[,i])*strip(y)[[i]]
		}
		return(r)
	}
	else {
		r <- .External(matrixmultiply(strip(x),xd[[1L]],xd[[2L]],y,yd[[1L]],yd[[2L]]))
		dim(r) <- c(xd[[1L]],yd[[2L]])
		return(r)
	}
}

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

