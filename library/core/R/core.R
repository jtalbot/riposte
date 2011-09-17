
c <- function(...) .Internal(unlist(list(...)))

ifelse <- function(test, yes, no) {
	if(!any(test)) no
	else {
		tmp <- yes
		if(!all(test)) yes[!test] <- no
		tmp
	}
}

`attr` <- function(x, which, exact=FALSE) {
	.Internal(attr(x, which, exact))
}

`attr<-` <- function(x, which, value) {
	.Internal(`attr<-`(x, which, value))
}

`rm` <- function(...) {
	.Internal(remove(list(...)))
}

#`[` <- function(x, ..., drop = TRUE) {
#	i <- list(...)
#	d <- dim(x)
#	i[is.na(i)] <- lapply(d[is.na(i)], function(x) 1:x)
#	d <- cumprod(c(1,d[-length(d)]))
#	i <- mapply(function(i) (i[[1]]-1)*i[[2]], i, d)
#	nd <- as.integer(lapply(i, function(x) length(x)))
#	len <- prod(nd)
#	a <- cumprod(nd)
#}

`[[` <- function(x, ..., exact = TRUE) {
	i <- as.integer(list(...))
	d <- dim(x)
	if(length(i) != length(d)) stop("incorrect number of subscripts")
	if(any(i < 1) || any(i > d)) stop("subscript out of bounds")
	d <- c(1,d[-length(d)])
	x[[sum((i-1)*cumprod(d))+1]]
}

