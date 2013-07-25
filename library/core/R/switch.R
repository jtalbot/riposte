
`switch` <- function(EXPR, ...) {
    p <- substitute(switch(EXPR, ...), environment(NULL))
    promise('r', p, parent.frame(1L), environment(NULL))
    r
}

