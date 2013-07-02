
as.vector <- function(x, mode) {
	switch(mode,
		logical = as(strip(x),"logical"),
		integer = as(strip(x),"integer"),
		double  = as(strip(x),"double"),
		numeric = as(strip(x),"double"),
		complex = as.complex(x),
		character = as(strip(x),"character"),
        list = {
            if(class(x) == 'name')
                list(x)
            else if(class(x) == 'expression' || class(x) == 'call')
                strip(x)
            else
                as(x, 'list') 
            },
		any = strip(x))
}

as.function.default <- function(x, envir) {
    call <- list()
    call[[1]] <- 'function'
    call[[2]] <- x[-length(x)]
    call[[3]] <- x[[length(x)]]
    call[[4]] <- deparse(call[[3]])
    class(call) <- 'call'
    promise('p', call, envir, core::parent.frame(0))
    return(p)
}
