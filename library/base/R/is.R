
is.null <- function(x) .Internal(typeof)(x) == "NULL"
is.logical <- function(x) .Internal(typeof)(x) == "logical"
is.integer <- function(x) .Internal(typeof)(x) == "integer"
is.real <- function(x) .Internal(typeof)(x) == "double"
is.double <- function(x) .Internal(typeof)(x) == "double"
is.complex <- function(x) .Internal(typeof)(x) == "complex"
is.character <- function(x) .Internal(typeof)(x) == "character"
is.symbol <- function(x) .Internal(typeof)(x) == "symbol"
is.environment <- function(x) .Internal(typeof)(x) == "environment"
is.list <- function(x) .Internal(typeof)(x) == "list"
is.pairlist <- function(x) .Internal(typeof)(x) == "pairlist"
is.expression <- function(x) .Internal(typeof)(x) == "expression"
is.raw <- function(x) .Internal(typeof)(x) == "raw"
is.call <- function(x) .Internal(typeof)(x) == "language" 
is.primitive <- function(x) .Internal(typeof)(x) == "builtin"

is.object <- function(x) "NYI"

is.numeric <- function(x) is.double(x) || is.integer(x)    #should also dispatch generic
is.matrix <- function(x) "NYI"
is.array <- function(x) "NYI"

is.atomic <- function(x) .Internal(switch)(typeof(x), logical=,integer=,double=,complex=,character=,raw=,NULL=TRUE,FALSE)
is.recursive <- function(x) !(is.atomic(x) || is.symbol(x))

is.language <- function(x) is.call(x) || is.environment(x) || is.symbol(x)
is.function <- function(x) .Internal(typeof)(x) == "function"

is.single <- function(x) "NYI"

is.na <- function(x) .Internal(is.na)(x)
is.nan <- function(x) .Internal(is.nan)(x)
is.finite <- function(x) .Internal(is.finite)(x)
is.infinite <- function(x) .Internal(is.infinite)(x)

is.vector <- function(x, mode="any") .Internal(switch)(mode,
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
