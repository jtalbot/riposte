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

logical <- function(length = 0L) vector("logical", length)
integer <- function(length = 0L) vector("integer", length)
double <- function(length = 0L) vector("double", length)
numeric <- double
real <- double
character <- function(length = 0L) vector("character", length)
raw <- function(length = 0L) vector("raw", length)

as.vector <- function(x, mode = "any") {
	switch(mode,
		logical = .Internal(as.logical(x)),
		integer = .Internal(as.integer(x)),
		double  = .Internal(as.double(x)),
		numeric = .Internal(as.double(x)),
		complex = as.complex(x),
		character = .Internal(as.character(x)),
		any = x)
}

cbind <- function(...) {
	l <- list(...)
	rows <- max(unlist(lapply(l, length)))
	matrix <- unlist(l)
	dim(matrix) <- c(rows, length(l))
	matrix
}
