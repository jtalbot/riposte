
max <- function(..., na.rm=FALSE) {
	x <- c(...)
	if(na.rm)	
		max(x[!is.na(x)])
	else
		max(x)
}

min <- function(..., na.rm=FALSE) {
	x <- c(...)
	if(na.rm)
		min(x[!is.na(x)])
	else
		min(x)
}

sum <- function(..., na.rm=FALSE) {
	x <- c(...)
	if(na.rm)
		sum(x[!is.na(x)])
	else
		sum(x)
}

prod <- function(..., na.rm=FALSE) {
	x <- c(...)
	if(na.rm)
		prod(x[!is.na(x)])
	else
		prod(x)
}

all <- function(..., na.rm=FALSE) {
	x <- c(...)
	if(na.rm)
		all(x[!is.na(x)])
	else
		all(x)
}

any <- function(..., na.rm=FALSE) {
	x <- c(...)
	if(na.rm)
		any(x[!is.na(x)])
	else
		any(x)
}

