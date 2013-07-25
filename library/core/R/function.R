
`function` <- function(arglist, expr)
{
    p <- substitute(function(arglist) expr)
    promise('r', p, parent.frame(1L), environment(NULL))
    r
}

`return` <- function(value)
{
    p <- substitute(return(value))
    promise('r', p, parent.frame(1L), environment(NULL))
    r
}

