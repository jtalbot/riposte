pi <- 3.1415926535897932384626433832795

#nargs <- function() { length(.Internal(sys.call(1L)))-1L }

options <- function(...) {}

factor <- function(x, levels) {
	attr(x, 'levels') <- levels
	x
}

split <- function(x, f) {
	split(strip(x), strip(f), length(attr(f, 'levels')))
}

c <- function(...) .Internal(unlist(list(...), TRUE, TRUE))
print <- function(...) cat(...)

`:` <- function(from, to) { 
	if(to > from) seq(from,1L,to-from+1L)
	else if(to < from) seq(from,-1L,from-to+1L)
	else seq(from,0L,1L)
}

dispatch1 <- function(op, x, default) {
	fun <- .Internal(paste(c(op, '.', class(x)), ""))
	if(exists(fun)) {
		get(fun)(x)
	}
	else {
		default(x)
	}
}

dispatch2 <- function(op, x, y, default) {
	funx <- .Internal(paste(c(op, '.', class(x)), ""))
	if(exists(funx)) {
		return(get(funx)(x,y))
	}
	funy <- .Internal(paste(c(op, '.', class(y)), ""))
	if(exists(funy)) {
		return(get(funy)(x,y))
	}
	default(x,y)
}

`+` <- function(x,y) {
	if(missing(y)) {
		dispatch1('+', x, function(x) x)
	} else {
		#dispatch2('+', x, y, function(x,y) strip(x)+strip(y))
		strip(x)+strip(y)
	}
}

`-` <- function(x,y) {
	if(missing(y)) {
		dispatch1('-', x, function(x) -strip(x))
	} else {
		dispatch2('-', x, y, function(x,y) strip(x)-strip(y))
	}
}

`*` <- function(x,y) {
	dispatch2('*', x, y, function(x,y) strip(x)*strip(y))
}

`/` <- function(x,y) {
	dispatch2('/', x, y, function(x,y) strip(x)*strip(y))
}

`<=` <- function(x,y) {
	dispatch2('<=', x, y, function(x,y) strip(x)<=strip(y))
}

`<` <- function(x,y) {
	dispatch2('<', x, y, function(x,y) strip(x)<strip(y))
}

`sqrt` <- function(x) {
	dispatch1('sqrt', x, function(x) sqrt(strip(x)))
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

`[` <- function(x, i, j) {
	if(nargs() == 2L || nargs() == -1L) {
		strip(x)[strip(i)]
	} else {
		d <- dim(x)
		if(missing(i) && missing(j))
			x
		else if(missing(i))
			strip(x)[(1L:d[[1L]])+(d[[1L]]*(as.integer(strip(j))-1L))]
		else if(missing(j))
			strip(x)[(0L:(d[[2L]]-1L))*(d[[1L]])+as.integer(strip(i))]
		else
			strip(x)[(as.integer(strip(i))-1L)*d[[1L]]+as.integer(strip(j))]	
	}
}

`[<-` <- function(x, i, ..., value) `[<-`(strip(x), strip(i), strip(value))
`[[<-` <- function(x, i, ..., value) `[[<-`(strip(x), strip(i), strip(value))

#`[[` <- function(x, ..., exact = TRUE) UseMethod('[[')

#`[[` <- function(x, ..., exact = TRUE) {
#	i <- as.integer(list(...))
#	
#	d <- dim(x)
#	if(is.null(d)) d <- length(x)
#	
#	if(length(i) != length(d)) stop("incorrect number of subscripts")
#	if(any(i < 1) || any(i > d)) stop("subscript out of bounds")
#
#	if(length(d) > 1) {
#		d <- c(1,d[-length(d)])
#		d <- sum((i-1)*cumprod(d))+1
#	}
#	strip(x)[[d]]
#}

`[[` <- function(x, i) {
	strip(x)[[strip(i)]]
}

length <- function(x) length(strip(x))

nrow <- function(x) dim(x)[1L]
ncol <- function(x) dim(x)[2L]

lapply <- function(x, func) {
	# should check that input is actually a list
	if(is.character(func)) {
		if(func == "sum") {
			return(sum(x))
		}
		else if(func == "prod") {
			return(prod(x))
		}
		else if(func == "mean") {
			return(mean(x))
		}
		else if(func == "length") {
			return(length(x))
		}
		else if(func == "min") {
			return(min(x))
		}
		else if(func == "max") {
			return(max(x))
		}
	}
	.Internal(mapply(list(x), func))
}

mapply <- function(FUN, ...) {
	.Internal(mapply(list(...), FUN))
}
