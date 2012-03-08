#nchar <- function(x) .Internal(nchar(x))
#nzchar <- function(x) .Internal(nzchar(x))

eigen <- function(x, symmetric=FALSE) {
	xd <- dim(x)
	if(is.null(xd)) xd <- c(1, length(x))
	if(symmetric)
		r <- .Internal(eigen.symmetric(strip(x), xd[1], xd[2]))
	else
		r <- .Internal(eigen(strip(x), xd[1], xd[2]))
	vec <- r[[2]]
	dim(vec) <- xd
	list(values=r[[1]], vectors=vec)
}

cat <- function(...) .Internal(cat(list(...)))
library <- function(.) .Internal(library(.))
#inherits <- function(x, what, which=FALSE) .Internal(inherits(x, what, which))

unlist <- function(x, recursive = TRUE, use.names = TRUE) UseMethod("unlist")
unlist.default <- function(x, recursive = TRUE, use.names = TRUE) {
	x <- .Internal(unlist(x, as.logical(recursive), as.logical(use.names)))
}

eval <- function(expr, envir=parent.frame()) .Internal(eval(expr, envir, NULL))
evalq <- function(expr, envir=parent.frame()) .Internal(eval(quote(expr), envir, NULL))
eval.parent <- function(expr, n=1) .Internal(eval(expr, parent.frame(n+1), NULL))
local <- function(expr, envir=new.env()) .Internal(eval(expr, envir, NULL))

source <- function(x) .Internal(source(x))

lapply <- function(x, func) .Internal(lapply(x, func))

parent.frame <- function(n=1) .Internal(parent.frame(n+1))
environment <- function(x=NULL) if(is.null(x)) parent.frame() else .Internal(environment(x))
new.env <- function() .Internal(new.env())
sys.call <- function(which=0) .Internal(sys.call(which+1))
alist <- function(...) as.list(sys.call())[-1L]
rm <- function(...) .Internal(remove(as.character(sys.call())[-1L], parent.frame()))

stop <- function(x) .Internal(stop(x))
warning <- function(x) .Internal(warning(x))

deparse <- function(x) .Internal(deparse(x))
substitute <- function(x) .Internal(substitute(x))

typeof <- function(x) .Internal(typeof(x))

exists <- function(x, envir, inherits=TRUE) {
	if(missing(envir)) envir <- parent.frame()
	.Internal(exists(x, envir, inherits, NULL))
}

proc.time <- function(x) .Internal(proc.time())
trace.config <- function(trace=0) .Internal(trace.config(trace))

read.table <- function(file) .Internal(read.table(file))
