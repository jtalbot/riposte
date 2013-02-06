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
library <- function(.) .External(library(.))
inherits <- function(x, what, which=FALSE) {
    #.External(inherits(x, what, which))
    any(class(x) == what)
}

#unlist <- function(x, recursive = TRUE, use.names = TRUE) UseMethod("unlist")
unlist <- function(x, recursive = TRUE, use.names = TRUE) {
	x <- .External(unlist(x, as.logical(recursive), as.logical(use.names)))
}

eval <- function(expr, envir=parent.frame()) .External(eval(expr, envir, NULL))
evalq <- function(expr, envir=parent.frame()) .External(eval(quote(expr), envir, NULL))
eval.parent <- function(expr, n=1) .External(eval(expr, parentframe(n+1), NULL))
local <- function(expr, envir=new.env()) .External(eval(expr, envir, NULL))

source <- function(x) .External(source(x))

parent.frame <- function(n=1) .External(parentframe(n+1))
environment <- function(x=NULL) if(is.null(x)) parent.frame() else .External(environment(x))
new.env <- function() .External(new.env())
sys.call <- function(which=0) .External(sys.call(which+1))
alist <- function(...) as.list(sys.call())[-1L]
#rm <- function(...) .External(remove(as.character(sys.call())[-1L], parent.frame()))

stop <- function(x) .External(stop(x))
warning <- function(x) .External(warning(x))

deparse <- function(x) .External(deparse(x))
substitute <- function(x) .External(substitute(x))

typeof <- function(x) .External(typeof(x))

exists <- function(x, envir, inherits=TRUE) {
	if(missing(envir)) envir <- parent.frame()
	.External(exists(x, envir, inherits, NULL))
}

get <- function(x, envir, inherits=TRUE) {
	if(missing(envir)) envir <- parent.frame()
	.External(get(x, envir, inherits, NULL))
}

proc.time <- function(x) .External(proc.time())
trace.config <- function(trace=0) .External(trace.config(trace))

read.table <- function(file,sep=" ",colClasses=c("double")) .External(readtable(file,sep,colClasses))

match <- function(x, table, nomatch = NA_integer_) {
	r <- .External(match(x, table))
	r[is.na(r)] <- nomatch
}

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
