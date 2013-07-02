
vector <- function(mode, length) vector(mode, length)

is.vector <- function(x, mode="any") 
    switch(mode,
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

