
options <- function(...) {}

factor <- function(x, levels) {
	attr(x, 'levels') <- levels
	x
}

split <- function(x, f) {
	split(strip(x), strip(f), length(attr(f, 'levels')))
}

`:` <- function(from, to) { 
	if(to > from) seq(from,1L,to-from+1L)
	else if(to < from) seq(from,-1L,from-to+1L)
	else seq(from,0L,1L)
}

dispatch1 <- function(op, x, default) {
	fun <- .External(paste(c(op, '.', class(x)), ""))
	if(exists(fun)) {
		get(fun)(x)
	}
	else {
		default(x)
	}
}

dispatch2 <- function(op, x, y, default) {
	funx <- .External(paste(c(op, '.', class(x)), ""))
	if(exists(funx)) {
		return(get(funx)(x,y))
	}
	funy <- .External(paste(c(op, '.', class(y)), ""))
	if(exists(funy)) {
		return(get(funy)(x,y))
	}
	default(x,y)
}

`+` <- function(x,y) {
	if(missing(y)) {
		dispatch1('+', x, function(x) x)
	} else {
		dispatch2('+', x, y, function(x,y) strip(x)+strip(y))
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

`[` <- function(x, ...) UseMethod('[', x, ...)

`[.default` <- function(x, ...) {
    if(nargs() < 2L) {
        x
    } else if(nargs() == 2L) {
        i <- ..1

        if(is.character(i)) {
            i <- which(names(x) == i)
        }

        r <- strip(x)[strip(i)]
        # discard all attributes but e.g. names
        if(!is.null(names(x))) {
            names(r) <- names(x)[strip(i)]
        }
        r
    } else if(nargs() == 3L) {
        d <- dim(x)
        if(missing(..1) && missing(..2))
            x
        else if(missing(..1))
            strip(x)[(1L:d[[1L]])+(d[[1L]]*(as.integer(strip(..2))-1L))]
        else if(missing(..2))
            strip(x)[(0L:(d[[2L]]-1L))*(d[[1L]])+as.integer(strip(..1))]
        else
            strip(x)[(as.integer(strip(..2))-1L)*d[[1L]]+as.integer(strip(..1))]	
    }
    else {
        stop("Unsupported indexing length")
    }
}

`[<-` <- function(x, ..., value) UseMethod('[<-', x, ..., value=value)

.copy.most.attributes <- function(r, x, i) {
    i <- strip(i)
    # copy over attributes, taking care to keep names lined up
    for(n in names(attributes(x))) {
        a <- attr(x,n)
        if(n == 'names') {
            a[i] <- ifelse(is.na(a[i]), '', a[i])
            a[is.na(a)] <- ''
        }
        attr(r,n) <- a
    }
    r
}

`[<-.default` <- function(x, i, ..., value) {
    r <- `[<-`(strip(x), strip(i), strip(value))
    .copy.most.attributes(r, x, i)
}

`[<-.list` <- function(x, i, value) {
    r <- `[<-`(strip(x), strip(i), value)
    .copy.most.attributes(r, x, i)
}

`[<-.call` <- `[<-.list`
`[<-.expression` <- `[<-.list`

`[[<-` <- function(x, ..., value) {
    UseMethod('[[<-', x)
}

`[[<-.default` <- function(x, i, ..., value) {
    if(is.character(i)) {
        i <- which(names(x) == i)
        if(length(i)==0)
            stop("subscript out of bounds") 
        else
            r <- `[[<-`(strip(x), strip(i[[1]]), strip(value))
    }
    else {
        r <- `[[<-`(strip(x), strip(i), strip(value))
    }
    .copy.most.attributes(r, x, i)
}

`[[<-.list` <- function(x, i, ..., value) {
    if(is.character(i)) {
        i <- which(names(x) == i)
        if(length(i)==0)
            stop("subscript out of bounds") 
        else
            r <- `[[<-`(strip(x), strip(i[[1]]), value)
    }
    else {
        r <- `[[<-`(strip(x), strip(i), value)
    }
    .copy.most.attributes(r, x, i)
}

`[[<-.call` <- `[[<-.list`
`[[<-.expression` <- `[[<-.list`

`[[` <- function(x, ..., exact = TRUE)
    UseMethod('[[', x, ..., exact=exact)

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

`[[.default` <- function(x, i, ..., exact = TRUE) {
    if(is.character(i)) {
        i <- which(names(x)==i)
        if(length(i)==0)
            stop("subscript out of bounds") 
        else
            strip(x)[[ i[[1]] ]]
    }
    else {
	    strip(x)[[strip(i)]]
    }
}

`[[.environment` <- function(x, i, ...) {
    strip(x)[[strip(i)]]
}

which <- function(x) {
    if(!is.logical(x))
        stop("argument to 'which' is not logical")

    seq(1,1,length(x))[x]
}

length <- function(x) length(strip(x))

nrow <- function(x) dim(x)[1L]
ncol <- function(x) dim(x)[2L]

lapply <- function(x, func) {
	.External(mapply(list(x), func))
}

mapply <- function(FUN, ...) {
	.External(mapply(list(...), FUN))
}

`$` <- function(a, b) {
    a[[strip(.pr_expr(parent.frame(0), quote(b)))]]
}

`$<-` <- function(x, i, value) {
    x[[strip(.pr_expr(parent.frame(0), quote(i)))]] <- value
    x
}

`::` <- function(a, b) {
    getNamespace(strip(.pr_expr(parent.frame(0), quote(a))))[[strip(.pr_expr(parent.frame(0), quote(b)))]]
}

