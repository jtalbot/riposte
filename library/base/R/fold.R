
max <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(max)(unlist(lapply(args, function(x) .Internal(max)(x[!is.na(x)]))))
	else
		.Internal(max)(unlist(lapply(args, .Internal(max))))
}

min <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(min)(unlist(lapply(args, function(x) .Internal(min)(x[!is.na(x)]))))
	else
		.Internal(min)(unlist(lapply(args, .Internal(min))))
}

sum <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(sum)(unlist(lapply(args, function(x) .Internal(sum)(x[!is.na(x)]))))
	else
		.Internal(sum)(unlist(lapply(args, .Internal(sum))))
}

prod <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(prod)(unlist(lapply(args, function(x) .Internal(prod)(x[!is.na(x)]))))
	else
		.Internal(prod)(unlist(lapply(args, .Internal(prod))))
}

all <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(all)(unlist(lapply(args, function(x) .Internal(all)(x[!is.na(x)]))))
	else
		.Internal(all)(unlist(lapply(args, .Internal(all))))
}

any <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(any)(unlist(lapply(args, function(x) .Internal(any)(x[!is.na(x)]))))
	else
		.Internal(any)(unlist(lapply(args, .Internal(any))))
}

