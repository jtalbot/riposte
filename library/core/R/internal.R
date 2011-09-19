nchar <- function(x) .Internal(nchar(x))
nzchar <- function(x) .Internal(nzchar(x))
is.na <- function(x) .Internal(is.na(x))
is.nan <- function(x) .Internal(is.nan(x))
is.finite <- function(x) .Internal(is.finite(x))
is.infinite <- function(x) .Internal(is.infinite(x))

cat <- function(...) .Internal(cat(list(...)))
library <- function(.) .Internal(library(.))
inherits <- function(x, what, which=FALSE) .Internal(inherits(x, what, which))

seq <- function(from=1, by=1, length.out=1) .Internal(seq(from, by, length.out))
rep <- function(v, each=1, length.out=1) .Internal(rep(v, each, length.out))

unlist <- function(x) .Internal(unlist(x))

eval <- function(expr, envir=parent.frame()) .Internal(eval(x, envir, NULL))
source <- function(x) .Internal(source(x))

lapply <- function(x, func) .Internal(lapply(x, func))

environment <- function(x) .Internal(environment(x))
parent.frame <- function(x) .Internal(parent.frame(x))

stop <- function(x) .Internal(stop(x))
warning <- function(x) .Internal(warning(x))

deparse <- function(x) .Internal(deparse(x))
substitute <- function(x) .Internal(substitute(x))

typeof <- function(x) .Internal(typeof(x))

exists <- function(x) .Internal(exists(x, NULL, NULL, NULL))

proc.time <- function(x) .Internal(proc.time())
trace.config <- function(trace=TRUE, verbose=FALSE) .Internal(trace.config(trace, verbose))
