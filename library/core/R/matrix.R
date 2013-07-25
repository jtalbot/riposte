
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

`%o%` <- function(x,y) {
	outer(x,y,`*`)
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

matrix <- function(data = NA, nrow = 1, ncol = 1) {
	if(length(data) < nrow*ncol)
		data <- rep(data, length.out=nrow*ncol)
	dim(data) <- c(nrow, ncol)
	class(data) <- 'matrix'
	data
}

outer <- function (X, Y, FUN = "*", ...) 
{
    if (is.array(X)) {
        dX <- dim(X)
        nx <- dimnames(X)
        no.nx <- is.null(nx)
    }
    else {
        dX <- length(X)
        no.nx <- is.null(names(X))
        if (!no.nx) 
            nx <- list(names(X))
    }
    if (is.array(Y)) {
        dY <- dim(Y)
        ny <- dimnames(Y)
        no.ny <- is.null(ny)
    }
    else {
        dY <- length(Y)
        no.ny <- is.null(names(Y))
        if (!no.ny) 
            ny <- list(names(Y))
    }
    if (is.character(FUN) && FUN == "*") {
        robj <- as.vector(X) %*% t(as.vector(Y))
        dim(robj) <- c(dX, dY)
	class(robj) <- 'matrix'
    }
    else {
        #FUN <- match.fun(FUN)
        Y <- rep.int(Y, rep.int(length(X), length(Y)))
        if (length(X)) 
            X <- rep(X, times = ceiling(length(Y)/length(X)))
        robj <- FUN(X, Y, ...)
        dim(robj) <- c(dX, dY)
	class(robj) <- 'matrix'
    }
    #if (no.nx) 
    #    nx <- vector("list", length(dX))
    #else if (no.ny) 
    #    ny <- vector("list", length(dY))
    #if (!(no.nx && no.ny)) 
    #    dimnames(robj) <- c(nx, ny)
    robj
}

upper.tri <- function(x, diag=FALSE) {
	if(diag)
		row(x) <= col(x)
	else
		row(x) < col(x)
}

row <- function(x) {
	xd <- dim(x)
	matrix(index(xd[[1L]], 1L, xd[[1L]]*xd[[2L]]), xd[[1L]], xd[[2L]])
}

col <- function(x) {
	xd <- dim(x)
	matrix(index(xd[[2L]], xd[[1L]], xd[[1L]]*xd[[2L]]), xd[[1L]], xd[[2L]])
}

`diag<-` <- function(x, value) {
	xd <- dim(x)
	len <- min(xd)
	r <- strip(x)
	r[(seq_len(len)-1L)*xd[[1L]]+seq_len(len)] <- value
	matrix(r, xd[[1L]], xd[[2L]])
}
