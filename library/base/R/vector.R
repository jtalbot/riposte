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

cbind <- function(...) {
	l <- list(...)
	rows <- max(unlist(lapply(l, length)))
	matrix <- unlist(l)
	dim(matrix) <- c(rows, length(l))
	matrix
}
