
force <- function(x) .External(force(x))

list <- function(...) list(...)

system.time <- function(expr) {
	start <- .External(proctime())
	.External(force(expr))
	.External(proctime())-start
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
`names<-` <- function(x, value) { 
    if(is.null(value))
        attr(x, 'names') <- NULL
    else {
        if(length(value) != length(x))
            stop("'names' attributes must be the same length as the vector")
        attr(x, 'names') <- as.character(value)
    }
    x 
}

dim <- function(x) attr(x, 'dim')
`dim<-` <- function(x, value) {
    if(is.null(value))
        attr(x, 'dim') <- NULL
    else {
    	if(length(value) == 0L) stop("length-0 dimension vector is invalid")
    	if(any(is.na(value))) stop("the dims contain missing values")
    	value <- as.integer(value)
    	if(any(value < 0L)) stop("the dims contain negative values")
    	if(prod(value) != length(x)) stop("dims product do not match the length of object")
    	attr(x, 'dim') <- as.integer(value)
    }
    x
}

class <- function(x) {
    r <- attr(x, 'class')
    if(is.null(r)) {
        r <- .type(x)
        if(r == 'double')
            r <- 'numeric'
    }
    r
}

`class<-` <- function(x, value) {
    if(is.null(value))
        attr(x, 'class') <- NULL
    else
        attr(x, 'class') <- as.character(value)
    x
}

oldClass <- function(x) attr(x, 'class')
`oldClass<-` <- function(x, value) attr(x, 'class') <- as.character(value)

dimnames <- function(x) attr(x, 'dimnames')
`dimnames<-` <- function(x, value) {
    #NYI: check length and dim
    attr(x, 'dimnames') <- value
    x
} 

#seq <- function(from=1, by=1, length.out=1) .External(seq(from, by, length.out))

rep <- function(x, times=1, length.out=times*each*length(x), each=1) {
	x[index(length(x), strip(each), strip(length.out))]
}

rep.int <- function(x, times) {
	times <- as.integer(times)
	if(length(times) == length(x))
		x[.External(repeat2(times, sum(times)))]
	else
		x[index(length(x), 1, times*length(x))]
}

attributes <- function(x) {
    .External(getAttributes(x))
}
