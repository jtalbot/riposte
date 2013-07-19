#nchar <- function(x) .External(nchar(x))
#nzchar <- function(x) .External(nzchar(x))

sort <- function(x) .External(sort(x))

eigen <- function(x, symmetric=FALSE) {
	xd <- dim(x)
	if(is.null(xd)) xd <- c(1, length(x))
	if(symmetric)
		r <- .External(eigensymmetric(strip(x), xd[1], xd[2]))
	else
		r <- .External(eigen(strip(x), xd[1], xd[2]))
	vec <- r[[2]]
	dim(vec) <- xd
	list(values=r[[1]], vectors=vec)
}

cat <- function(..., sep = " ") .External(cat(list(...), sep))

library <- function(.) {
    e <- .env_new(getRegisteredNamespace('core'))
    .External(library(e, .))
    setRegisteredNamespace(., e)
    e
}

attach <- function(.) {
    g <- globalenv()
    e <- .env_new(environment(g))
    # copy over exported bindings
    for(n in .env_names(.)) {
        e[[n]] <- .[[n]]
    }
    environment(g) <- e
}

inherits <- function(x, what, which=FALSE) {
    #.External(inherits(x, what, which))
    any(class(x) == what)
}

source <- function(x) .External(source(x))

sys.call <- function(which = 0L) {
    .frame(as.integer(-which+1L))[[2L]]
}

sys.frame <- function(which = 0L) {
    .frame(as.integer(-which+1L))[[1L]]
}

sys.nframe <- function() {
    0
}

sys.function <- function(which = 0L) {
    .frame(as.integer(-which+1L))[[3L]]
}

sys.nargs <- function(which = 0L) {
    .frame(as.integer(-which+1L))[[4L]]
}

sys.parent <- function(n = 1L) {
    -n
}

parent.frame <- function(n = 1L) {
    .frame(as.integer(n+1L))[[1L]]
}

nargs <- function() {
    .frame(1L)[[4L]]
}

alist <- function(...) as.list(sys.call())[-1L]
#rm <- function(...) .External(remove(as.character(sys.call())[-1L], parent.frame()))

stop <- function(x) .External(stop(x))
warning <- function(x) .External(warning(x))

substitute <- function(x) .External(substitute(x))

exists <- function(x, envir, mode, inherits=TRUE) {
	if(missing(envir)) envir <- parent.frame(1)
	.External(exists(x, envir, inherits, NULL))
}

get <- function(x, envir, inherits=TRUE) {
	if(missing(envir)) envir <- parent.frame()
	.External(get(x, envir, inherits, NULL))
}

proc.time <- function(x) .External(proc.time())
trace.config <- function(trace=0) .External(trace.config(trace))

read.table <- function(file,sep=" ",colClasses=c("double")) .External(readtable(file,sep,colClasses))

commandArgs <- function (trailingOnly = FALSE) 
{
    args <- .External(commandArgs())
    if (trailingOnly) {
        m <- match("--args", args, 0L)
        if (m) 
            args[(m+1):length(args)]
        else character()
    }
    else args
}
