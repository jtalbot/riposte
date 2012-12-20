
as.null <- function(x,...) .Internal(as.null(x))
as.logical <- function(x,...) .Internal(as.logical(x))
as.integer <- function(x,...) .Internal(as.integer(x))
as.double <- function(x,...) .Internal(as.double(x))
as.character <- function(x,...) .Internal(as.character(x))
as.list <- function(x,...) .Internal(as.list(x))
as.numeric <- function(x,...) .Internal(as.double(x))

is.null <- function(x) .Internal(typeof(x)) == "NULL"
is.logical <- function(x) .Internal(typeof(x)) == "logical"
is.integer <- function(x) .Internal(typeof(x)) == "integer"
is.real <- function(x) .Internal(typeof(x)) == "double"
is.double <- function(x) .Internal(typeof(x)) == "double"
is.character <- function(x) .Internal(typeof(x)) == "character"
is.symbol <- function(x) .Internal(typeof(x)) == "symbol"
is.environment <- function(x) .Internal(typeof(x)) == "environment"
is.list <- function(x) .Internal(typeof(x)) == "list"
is.pairlist <- function(x) .Internal(typeof(x)) == "pairlist"
is.expression <- function(x) .Internal(typeof(x)) == "expression"
is.raw <- function(x) .Internal(typeof(x)) == "raw"
is.call <- function(x) .Internal(typeof(x)) == "language" 
is.primitive <- function(x) .Internal(typeof(x)) == "builtin"

is.object <- function(x) "NYI"

is.numeric <- function(x) is.double(x) || is.integer(x)    #should also dispatch generic
is.matrix <- function(x) is.numeric(dim(x))
is.array <- is.matrix

is.atomic <- function(x) switch(typeof(x), logical=,integer=,double=,complex=,character=,raw=,NULL=TRUE,FALSE)
is.recursive <- function(x) !(is.atomic(x) || is.symbol(x))

is.language <- function(x) is.call(x) || is.environment(x) || is.symbol(x)
is.function <- function(x) .Internal(typeof(x)) == "function"

is.single <- function(x) "NYI"

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

mode <- function(x)
	switch(typeof(x),
		integer="numeric",
		double="numeric",
		special="function",
		builtin="function",
		symbol="name",
		language="call",
		typeof(x))

`mode<-` <- function(x, value) {
	switch(value,
		logical=as.logical(x), 
		integer=as.integer(x),
		double=as.double(x),
		complex=as.complex(x),
		raw=as.raw(x),
		character=as.character(x),
		list=as.list(x),
		expression=as.expression(x),
		name=as.name(x),
		symbol=as.symbol(x), 
		error("unsupported mode"))
}

storage.mode <- function(x)
	switch(typeof(x),
		integer="integer",
		double="double",
		special="function",
		builtin="function",
		symbol="name",
		language="call",
		typeof(x))

`storage.mode<-` <- `mode<-`

as.environment <- function(x)
	switch(typeof(x),
		environment=x,
		double=,
		integer=parent.frame(2),
		error("unsupported cast to environment")) 

