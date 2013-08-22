
Re <- function(x) UseGroupMethod("Re", "Complex", x)
Re.default <- function(x) {
    .ArithCheckUnary(x)
    x
}
Re.complex <- function(x) {
    x[[1]]
}

Im <- function(x) UseGroupMethod("Im", "Complex", x)
Im.default <- function(x) {
    .ArithCheckUnary(x)
    vector('double', length(x))
}
Im.complex <- function(x) {
    x[[2]]
}

Mod <- function(x) UseGroupMethod("Mod", "Complex", x)
Mod.default <- function(x) {
    .ArithCheckUnary(x)
    x
}
Mod.complex <- function(x) {
    hypot(x[[2]], x[[1]])
}

Arg <- function(x) UseGroupMethod("Arg", "Complex", x)
Arg.default <- function(x) {
    .ArithCheckUnary(x)
    vector('double', length(x))
}
Arg.complex <- function(x) {
    atan2(x[[2]], x[[1]])
}

Conj <- function(x) UseGroupMethod("Conj", "Complex", x)
Conj.default <- function(x) {
    .ArithCheckUnary(x)
    x
}
Conj.complex <- function(x) {
    r <- list(Re=x[[1]], Im=-x[[2]])
    class(r) <- 'complex'
    r 
}

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
is.complex <- function(x) class(x) == "complex"

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
    re <- unlist(.Map(function(a) a[[1L]], list(l)))
    im <- unlist(.Map(function(a) a[[2L]], list(l)))
    r <- list(Re=re, Im=im)
    class(r) <- 'complex'
	r	
}

