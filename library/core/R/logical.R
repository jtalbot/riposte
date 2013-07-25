
as.logical <- function(x, ...) as(strip(x), "logical")

is.logical <- function(x) .type(x) == 'logical'

