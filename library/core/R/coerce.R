
is.raw <- function(x) .type(x) == "raw"

is.matrix <- function(x) is.numeric(dim(x))
is.array <- is.matrix

is.atomic <- function(x) switch(.type(x), logical=,integer=,double=,complex=,character=,raw=,NULL=TRUE,FALSE)
is.recursive <- function(x) !(is.atomic(x) || is.symbol(x))

is.language <- function(x) is.call(x) || is.environment(x) || is.symbol(x)

is.single <- function(x) .stop('type "single" unimplemented in R')

