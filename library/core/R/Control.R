
`if` <- function(cond, cons.expr, alt.expr=NULL)
{
    p <- substitute(if(cond) cons.expr else alt.expr)
    promise('r', p, .frame(1L), .getenv(NULL))
    r
}

`for` <- function(var, seq, expr)
{
    p <- substitute(for(var in seq) expr)
    promise('r', p, .frame(1L), .getenv(NULL))
    r
}

`while` <- function(cond, expr)
{
    p <- substitute(while(cond) expr)
    promise('r', p, .frame(1L), .getenv(NULL))
    r
}

`repeat` <- function(expr)
{
    p <- substitute(repeat expr)
    promise('r', p, .frame(1L), .getenv(NULL))
    r
}

`break` <- function()
{
    # NYI: break is compiled, not interpreted
    .stop("break (NYI)")
}

`next` <- function()
{
    # NYI: next is compiled, not interpreted
    .stop("next (NYI)")
}

