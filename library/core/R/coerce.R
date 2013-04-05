
as.null <- function(x,...) as(x,"NULL")
as.logical <- function(x,...) as(x,"logical")
as.integer <- function(x,...) as(x,"integer")
as.double <- function(x,...) as(x,"double")
as.character <- function(x,...) as(x,"character")
as.list <- function(x,...) as(x,"list")
as.raw <- function(x,...) as(x,"raw")
as.numeric <- function(x,...) as(x,"double")

is.null <- function(x) typeof(x) == "NULL"
is.logical <- function(x) typeof(x) == "logical"
is.integer <- function(x) typeof(x) == "integer"
is.real <- function(x) typeof(x) == "double"
is.double <- function(x) typeof(x) == "double"
is.character <- function(x) typeof(x) == "character"
is.environment <- function(x) typeof(x) == "environment"
is.list <- function(x) typeof(x) == "list"
is.raw <- function(x) typeof(x) == "raw"

is.symbol <- function(x) class(x) == "symbol"
is.pairlist <- function(x) class(x) == "pairlist"
is.expression <- function(x) class(x) == "expression"
is.call <- function(x) class(x) == "call" 
is.primitive <- function(x) class(x) == "builtin"

is.object <- function(x) !is.null(attr(x,'class'))

is.numeric <- function(x) is.double(x) || is.integer(x)    #should also dispatch generic
is.matrix <- function(x) is.numeric(dim(x))
is.array <- is.matrix

is.atomic <- function(x) switch(typeof(x), logical=,integer=,double=,complex=,character=,raw=,NULL=TRUE,FALSE)
is.recursive <- function(x) !(is.atomic(x) || is.symbol(x))

is.language <- function(x) is.call(x) || is.environment(x) || is.symbol(x)
is.function <- function(x) typeof(x) == "closure"

is.single <- function(x) stop('type "single" unimplemented in R')

is.vector <- function(x, mode="any") switch(mode,
	NULL=is.null(x),
	logical=is.logical(x),
	integer=is.integer(x),
	real=is.real(x),
	double=is.double(x),
	complex=is.complex(x),
	character=is.character(x),
	symbol=is.symbol(x),
	environment=is.environment(x),
	list=is.list(x),
	pairlist=is.pairlist(x),
	numeric=is.numeric(x),
	any=is.atomic(x) || is.list(x) || is.expression(x),
	FALSE)
# is.vector is also defined to check whether or not there are any attributes other than names(?!)

`storage.mode<-` <- function(x, value) {
	switch(value,
		logical=as(x,"logical"), 
		integer=as(x,"integer"),
		double=as(x,"double"),
		complex=as.complex(x),
		raw=as(x,"raw"),
		character=as(x,"character"),
		list=as(x,"list"),
		expression=as.expression(x),
		name=as.name(x),
		symbol=as.symbol(x), 
		stop("unsupported mode"))
}

.list2env <- function(l) {
    n <- names(l)
    if(length(l) > 0 && (!is.character(n) || length(n) != length(l)))
        stop("names(x) must be a character vector of the same length as x")

    e <- new.env(FALSE, emptyenv(), 0)
    for(i in seq(1, 1, length(l))) {
        e[[n[[i]]]] <- l[[i]]
    }
    e
}

as.environment <- function(x)
	switch(typeof(x),
		environment=x,
		double=,
		integer=parent.frame(x+1),
        list=.list2env(x),
		stop("unsupported cast to environment")) 

