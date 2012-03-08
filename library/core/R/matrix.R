
`%*%` <- function(x,y) {

	xd <- dim(x)
	if(is.null(xd)) xd <- c(1L, length(x))
	yd <- dim(y)
	if(is.null(yd)) yd <- c(length(y), 1L)

	if(xd[[2L]] != yd[[1L]])
		stop("Matrices are not conformable")

	if(xd[[1L]] == 1L && yd[[2L]] == 1L) {
		return(sum(strip(x)*strip(y)))
	} else if(xd[[1L]] == 1L) {
		r <- double(0)
		for(i in 1L:yd[[2L]])
			r[i] <- sum(strip(x)*strip(y[,i]))
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
		r <- .Internal(matrix.multiply(strip(x),xd[[1L]],xd[[2L]],y,yd[[1L]],yd[[2L]]))
		dim(r) <- c(xd[[1L]],yd[[2L]])
		return(r)
	}
}

matrix <- function(data = NA, nrow = 1, ncol = 1) {
	r <- rep(data, length.out=nrow*ncol)
	dim(r) <- c(nrow, ncol)
	r
}
