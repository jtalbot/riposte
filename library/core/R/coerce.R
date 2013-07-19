
as.null <- function(x,...) as(x,"NULL")
as.logical <- function(x,...) as(x,"logical")
as.integer <- function(x,...) as(x,"integer")
as.double <- function(x,...) as(x,"double")
as.character <- function(x,...) as(x,"character")
as.list <- function(x,...) as(x,"list")
as.raw <- function(x,...) as(x,"raw")
as.numeric <- function(x,...) as(x,"double")

is.null <- function(x) .type(x) == "NULL"
is.logical <- function(x) .type(x) == "logical"
is.integer <- function(x) .type(x) == "integer"
is.real <- function(x) .type(x) == "double"
is.double <- function(x) .type(x) == "double"
is.character <- function(x) .type(x) == "character"
is.environment <- function(x) .type(x) == "environment"
is.list <- function(x) .type(x) == "list"
is.raw <- function(x) .type(x) == "raw"

is.symbol <- function(x) class(x) == "name"
is.pairlist <- function(x) class(x) == "pairlist"
is.expression <- function(x) class(x) == "expression"
is.primitive <- function(x) class(x) == "builtin"

is.object <- function(x) !is.null(attr(x,'class'))

is.numeric <- function(x) is.double(x) || is.integer(x)    #should also dispatch generic
is.matrix <- function(x) is.numeric(dim(x))
is.array <- is.matrix

is.atomic <- function(x) switch(.type(x), logical=,integer=,double=,complex=,character=,raw=,NULL=TRUE,FALSE)
is.recursive <- function(x) !(is.atomic(x) || is.symbol(x))

is.language <- function(x) is.call(x) || is.environment(x) || is.symbol(x)
is.function <- function(x) .type(x) == "closure"

is.single <- function(x) stop('type "single" unimplemented in R')

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

    e <- .env_new(emptyenv())
    for(i in seq_len(length(l))) {
        e[[n[[i]]]] <- l[[i]]
    }
    e
}

.search.path <- function(n, env) {
    if(n > 0L) {
        env <- .GlobalEnv
        while(n > 0) {
            env <- environment(env)
            n <- n-1
        }
        env
    }
    else {
        environment(env)
    }
}

as.environment <- function(x)
	switch(.type(x),
		environment=x,
		double=,
		integer=.search.path(as.integer(x), parent.frame()),
        list=.list2env(x),
		stop("unsupported cast to environment")) 

as.name <- function(x) {
    x <- as.character(x)
    class(x) <- 'name'
    x
}
