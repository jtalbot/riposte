
expression <- function(...) {
    r <- list(...)
    class(r) <- 'expression'
    r
}

is.expression <- function(x) {
    inherits(x, 'expression', FALSE)
}

