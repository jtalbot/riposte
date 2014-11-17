
expression <- function(...) {
    r <- `.__call__.`[-1L]
    class(r) <- 'expression'
    r
}

is.expression <- function(x) {
    inherits(x, 'expression', FALSE)
}

