#  File src/library/base/R/vector.R
#  Part of the R package, http://www.R-project.org
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  A copy of the GNU General Public License is available at
#  http://www.r-project.org/Licenses/

vector <- function(mode = "logical", length = 0)
	switch(mode,
		logical=logical(length),
		integer=integer(length),
		double=double(length),
		numeric=double(length),
		character=character(length),
		complex=complex(length),
		raw=raw(length),
		stop("cannot make a vector of mode 'NYI'"))
real <- function(length) double(length)
numeric <- function(length) double(length)

# This function is currently not working since "complex" is interpreted as an op code.
complex <- function(length.out = 0,
		    real = numeric(), imaginary = numeric(),
		    modulus = 1, argument = 0) {
    if(missing(modulus) && missing(argument)) {
	## assume 'real' and 'imaginary'
	n <- max(length.out, length(real), length(imaginary))
	vector("complex", n) + real + imaginary*1i
    } else {
	n <- max(length.out, length(argument), length(modulus))
	rep(modulus,length.out=n) *
	    exp(1i * rep(argument, length.out=n))
    }
}

is.null <- function(x) .Internal(typeof)(x) == "NULL"
is.logical <- function(x) .Internal(typeof)(x) == "logical"
is.integer <- function(x) .Internal(typeof)(x) == "integer"
is.real <- function(x) .Internal(typeof)(x) == "double"
is.double <- function(x) .Internal(typeof)(x) == "double"
is.complex <- function(x) .Internal(typeof)(x) == "complex"
is.character <- function(x) .Internal(typeof)(x) == "character"
is.symbol <- function(x) .Internal(typeof)(x) == "symbol"
is.environment <- function(x) .Internal(typeof)(x) == "environment"
is.list <- function(x) .Internal(typeof)(x) == "list"
is.pairlist <- function(x) .Internal(typeof)(x) == "pairlist"
is.expression <- function(x) .Internal(typeof)(x) == "expression"
is.raw <- function(x) .Internal(typeof)(x) == "raw"
is.call <- function(x) .Internal(typeof)(x) == "language" 

is.object <- function(x) "NYI"

is.numeric <- function(x) is.double(x) || is.integer(x)    #should also dispatch generic
is.matrix <- function(x) "NYI"
is.array <- function(x) "NYI"

is.atomic <- function(x) .Internal(switch)(typeof(x), logical=,integer=,double=,complex=,character=,raw=,NULL=TRUE,FALSE)
is.recursive <- function(x) !(is.atomic(x) || is.symbol(x))

is.language <- function(x) is.call(x) || is.environment(x) || is.symbol(x)
is.function <- function(x) .Internal(typeof)(x) == "function"

is.single <- function(x) "NYI"

is.na <- function(x) .Internal(is.na)(x)
is.nan <- function(x) .Internal(is.nan)(x)
is.finite <- function(x) .Internal(is.finite)(x)
is.infinite <- function(x) .Internal(is.infinite)(x)

is.vector <- function(x, mode="any") .Internal(switch)(mode,
	NULL=is.null(x),
	logical=is.logical(x),
	integer=is.integer(x),
	real=is.real(x),
	double=is.double(x),
	complex=is.complex(x),
	character=is.character(x),
	symbol=is.symbol(x),
	environment=is.environment(x),
	list=is.list(x),
	pairlist=is.pairlist(x),
	numeric=is.numeric(x),
	any=is.atomic(x) || is.list(x) || is.expression(x),
	FALSE)
# is.vector is also defined to check whether or not there are any attributes other than names(?!)

as.vector <- function(x, mode = "any") {
	switch(mode,
		logical = .Internal(as.logical)(x),
		integer = .Internal(as.integer)(x),
		double  = .Internal(as.double)(x),
		numeric = .Internal(as.double)(x),
		complex = .Internal(as.complex)(x),
		character = .Internal(as.character)(x),
		any = x)
}


