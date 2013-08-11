
vector <- function(mode, length) vector(mode, length)

as.vector <- function(x, mode) {
	switch(mode,
		logical = as(strip(x),"logical"),
		integer = as(strip(x),"integer"),
		double  = as(strip(x),"double"),
		numeric = as(strip(x),"double"),
		complex = as.complex(x),
		character = as(strip(x),"character"),
        symbol = {
            x <- as.character.default(strip(x))[[1]]
            class(x) <- 'name'
            x
            },
        list = {
            if(class(x) == 'name')
                list(x)
            else if(class(x) == 'expression' || class(x) == 'call')
                x
            else {
                r <- as(strip(x), 'list')
                attributes(r) <- attributes(x)
                r
            }
            },
		any = strip(x))
}

is.vector <- function(x, mode) 
    switch(mode,
        NULL=is.null(x),
        logical=is.logical(x),
        integer=is.integer(x),
        real=is.real(x),
        double=is.double(x),
        complex=is.complex(x),
        character=is.character.default(x),
        symbol=is.symbol(x),
        environment=is.environment(x),
        list=is.list(x),
        pairlist=is.pairlist(x),
        numeric=is.numeric(x),
        any=is.atomic(x) || is.list(x),
        FALSE)
# is.vector is also defined to check whether or not there are any attributes other than names(?!)

