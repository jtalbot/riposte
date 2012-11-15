

#complex <- function(length.out = 0,
#		    real = numeric(), imaginary = numeric(),
#		    modulus = 1, argument = 0) {
#    if(missing(modulus) && missing(argument)) {
#	## assume 'real' and 'imaginary'
#	n <- max(length.out, length(real), length(imaginary))
#	vector("complex", n) + real + imaginary*1i
#    } else {
#	n <- max(length.out, length(argument), length(modulus))
#	rep(modulus,length.out=n) *
#	    exp(1i * rep(argument, length.out=n))
#    }
#}

complex <- function(length.out = 0L, real = numeric(), imaginary = numeric()) {
	if(missing(real) && missing(imaginary) && missing(modulus) && missing(argument)) {
		r <- list(Re=numeric(length.out), Im=numeric(length.out))
	} else if(missing(modulus) && missing(argument)) {
		r <- list(Re=real, Im=imaginary)
	} 
	class(r) <- 'complex'
	r
}

as.complex <- function(x,...) "NYI"
is.complex <- function(x) .Internal(typeof(x)) == "complex"

length.complex <- function(x) length(x[[1]])

`+.complex` <- function(x, y) {
	if(missing(y)) x
	else {
		if(inherits(x, 'complex') && inherits(y, 'complex'))
			r <- list(Re=x[[1]]+y[[1]], Im=x[[2]]+y[[2]])
		else if(inherits(x, 'complex'))
			r <- list(Re=x[[1]]+y, Im=x[[2]])
		else
			r <- list(Re=x+y[[1]], Im=y[[2]])
		class(r) <- 'complex'
		r
	}
}

`-.complex` <- function(x, y) {
	if(missing(y)) {
		r <- list(Re=-x[[1]], Im=-x[[2]])
	}
	else {
		if(inherits(x, 'complex') && inherits(y, 'complex'))
			r <- list(Re=x[[1]]-y[[1]], Im=x[[2]]-y[[2]])
		else if(inherits(x, 'complex'))
			r <- list(Re=x[[1]]-y, Im=x[[2]])
		else
			r <- list(Re=x-y[[1]], Im=y[[2]])
	}
	class(r) <- 'complex'
	r
}

`*.complex` <- function(x, y) {
	if(inherits(x, 'complex') && inherits(y, 'complex'))
		r <- list(Re=x[[1]]*y[[1]]-x[[2]]*y[[2]], Im=x[[2]]*y[[1]]+x[[1]]*y[[2]])
	else if(inherits(x, 'complex'))
		r <- list(Re=x[[1]]*y, Im=x[[2]]*y)
	else
		r <- list(Re=x*y[[1]], Im=x*y[[2]])

	class(r) <- 'complex'
	r
}

`/.complex` <- function(x, y) {
	if(inherits(x, 'complex') && inherits(y, 'complex')) {
		d <- y[[1]]*y[[1]] + y[[2]]*y[[2]]
		r <- list(Re=(x[[1]]*y[[1]]+x[[2]]*y[[2]])/d, Im=(x[[2]]*y[[1]]-x[[1]]*y[[2]])/d)
	}
	else if(inherits(x, 'complex'))
		r <- list(Re=x[[1]]/y, Im=x[[2]]/y)
	else {
		d <- y[[1]]*y[[1]] + y[[2]]*y[[2]]
		r <- list(Re=x*y[[1]]/d, Im=-x*y[[2]]/d)
	}
	class(r) <- 'complex'
	r
}

`c.complex` <- function(x, ...) {
    l <- list(x,...)
    re <- unlist(lapply(l, function(a) a[[1]]))
    im <- unlist(lapply(l, function(a) a[[2]]))
    r <- list(Re=re, Im=im)
    class(r) <- 'complex'
	r	
}

Re <- function(x) UseMethod("Re")
Im <- function(x) UseMethod("Im")

Re.numeric <- function(x) x
Im.numeric <- function(x) numeric(length(x))

Re.complex <- function(x) {
	x[[1]]
}

Im.complex <- function(x) {
	x[[2]]
}
