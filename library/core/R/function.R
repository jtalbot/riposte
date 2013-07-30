
`function` <- function(arglist, expr)
{
    p <- substitute(function(arglist) expr)
    print(p)
    promise('r', p, .frame(1L)[[1L]], .getenv(NULL))
    r
}

`return` <- function(value)
{
    p <- substitute(return(value))
    promise('r', p, .frame(1L)[[1L]], .getenv(NULL))
    r
}

