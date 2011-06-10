
max <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(max)(as.double(flatten(lapply(args, function(x) .Internal(max)(x[!is.na(x)])))))
	else
		.Internal(max)(as.double(flatten(lapply(args, .Internal(max)))))
}

min <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(min)(as.double(flatten(lapply(args, function(x) .Internal(min)(x[!is.na(x)])))))
	else
		.Internal(min)(as.double(flatten(lapply(args, .Internal(min)))))
}

sum <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(sum)(as.double(flatten(lapply(args, function(x) .Internal(sum)(x[!is.na(x)])))))
	else
		.Internal(sum)(as.double(flatten(lapply(args, .Internal(sum)))))
}

prod <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(prod)(as.double(flatten(lapply(args, function(x) .Internal(prod)(x[!is.na(x)])))))
	else
		.Internal(prod)(as.double(flatten(lapply(args, .Internal(prod)))))
}

all <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(all)(as.logical(flatten(lapply(args, function(x) .Internal(all)(x[!is.na(x)])))))
	else
		.Internal(all)(as.logical(flatten(lapply(args, .Internal(all)))))
}

any <- function(..., na.rm=FALSE) {
	args <- list(...)
	if(na.rm)
		.Internal(any)(as.logical(flatten(lapply(args, function(x) .Internal(any)(x[!is.na(x)])))))
	else
		.Internal(any)(as.logical(flatten(lapply(args, .Internal(any)))))
}

