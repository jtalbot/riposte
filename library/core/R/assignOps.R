
'=' <- `<-` <- function(x, value)
{
    p <- substitute(x <- value)
    promise('r', p, .frame(1L), .getenv(NULL))
    r
}

`<<-` <- function(x, value)
{
    p <- substitute(x <<- value)
    promise('r', p, .frame(1L), .getenv(NULL))
    r
}

`->` <- function(value, x)
{
    p <- substitute(x <- value)
    promise('r', p, .frame(1L), .getenv(NULL))
    r
}

`->>` <- function(value, x)
{
    p <- substitute(x <<- value)
    promise('r', p, .frame(1L), .getenv(NULL))
    r
}

