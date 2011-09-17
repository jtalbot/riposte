
force <- function(x) x

list <- function(...) list(...)

cat <- function(...) .Internal(cat(list(...)))

mapply <- function(FUN, ...) {
	lapply(t.list(...), FUN)
}

paste <- function(..., sep = " ", collapse = NULL) {
	r <- mapply(function(x) .Internal(paste(x, sep)), ...)
	if(!is.null(collapse)) .Internal(paste(r, collapse))
	else unlist(r)
}

anyDuplicated <- function(x) {
	for(i in seq_len(length(x)-1)) {
		for(j in (i+1):length(x)) {
			if(x[[i]] == x[[j]]) return(j)
		}
	}
	0 
}

make.names <- function(x) {
	x
}

names <- function(x) attr(x, 'names')
`names<-` <- function(x, value) attr(x, 'names') <- as.character(value)  #NYI: check length
dim <- function(x) attr(x, 'dim')
`dim<-` <- function(x, value) {
	if(length(value) == 0L) stop("length-0 dimension vector is invalid")
	if(any(is.na(value))) stop("the dims contain missing values")
	value <- as.integer(value)
	if(any(value < 0L)) stop("the dims contain negative values")
	if(prod(value) != length(x)) stop("dims product do not match the length of object")
	attr(x, 'dim') <- as.integer(value)
}
class <- function(x) attr(x, 'class')
`class<-` <- function(x, value) attr(x, 'class') <- as.character(value)
dimnames <- function(x) attr(x, 'dimnames')
`dimnames<-` <- function(x, value) attr(x, 'dimnames') <- value   #NYI: check length and dim
