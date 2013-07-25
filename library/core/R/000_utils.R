
.isTRUE <- function(x) {
    .type(x) == 'logical' && length(x) == 1L && !is.na(x) && x
}

.isFALSE <- function(x) {
    .type(x) == 'logical' && length(x) == 1L && !is.na(x) && !x
}

.c <- function(...) {
    unlist(list(...), FALSE, TRUE)
}

.characters <- function(...) {
    as.character(list(...))
}

.nargs <- function() {
    .frame(1L)[[4L]]
}

.export <- function(name, env, exported) {
    r <- .env_new(emptyenv())

    attr(r, 'name') <- name

    for(f in exported) {
        r[[f]] <- env[[f]]
    }

    r 
}
.make.namespace <- function(name, env, exported) {
    r <- .env_new(emptyenv())
    e <- .env_new(emptyenv())

    attr(r, 'name') <- name

    for(f in exported) {
        r[[f]] <- env[[f]]
        e[[f]] <- f
    }

    r[['.__NAMESPACE__.']] <- .env_new(emptyenv())
    r[['.__NAMESPACE__.']][['exports']] <- e 

    r 
}

.attach <- function(env) {
    .setenv(env, .getenv(.env_global()))
    .setenv(.env_global(), env)
    env
}

.cat <- function(..., sep="") .External(cat(list(...), sep))

.stop <- function(x) {
    n <- 1L
    repeat {
        call <- .frame(n)[[2L]]
        if(!is.null(call))
            .cat(n, ': ', format.call(call),'\n')

        if(is.null(.frame(n)[[6L]]))
            break
        if(n > 10L)
            break

        n <- n+1L
    }

    .cat('Error: ', x,'\n')

    .External(stop(x))
}

.escape <- function(x) {
    .Map.character("escape_map", as.character(x))
}

.pconcat <- function(x,y) {
    .Map.character("concat_map", as.character(x), as.character(y))
}

.concat <- function(x) {
    .Fold.character("concat", as.character(x))
}

# argument check functions...
.ArithCheckUnary <- function(x) {
    switch(.type(x),
        double=TRUE,
        integer=TRUE,
        logical=TRUE,
        .stop("non-numeric argument to mathematical function"))
}

.ArithCheckBinary <- function(x,y) {
    switch(.type(x),
        double=TRUE,
        integer=TRUE,
        logical=TRUE,
        .stop("non-numeric argument to mathematical function"))
    switch(.type(y),
        double=TRUE,
        integer=TRUE,
        logical=TRUE,
        .stop("non-numeric argument to mathematical function"))
}

.OrdinalCheckBinary <- function(x,y) {
    if(!is.atomic(x) && !is.list(x))
        .stop("comparison is possible only for atomic and list types")
    if(!is.atomic(y) && !is.list(y))
        .stop("comparison is possible only for atomic and list types")
}

.LogicCheckUnary <- function(x) {
    switch(.type(x),
        double=TRUE,
        integer=TRUE,
        logical=TRUE,
        .stop("invalid argument type"))
}

.LogicCheckBinary <- function(x,y) {
    switch(.type(x),
        double=TRUE,
        integer=TRUE,
        logical=TRUE,
        .stop("operations are only possible for numeric, logical or complex types"))
    switch(.type(y),
        double=TRUE,
        integer=TRUE,
        logical=TRUE,
        .stop("operations are only possible for numeric, logical or complex types"))
}

# math dispatch functions...

.ArithUnary1 <- function(ffunc, ifunc, x) {
    switch(.type(x),
        double=.Map.double(ffunc, x),
        integer=,
        logical=.Map.integer(ifunc, as.integer(x)),
        .stop("non-numeric argument to mathematical function"))
}

.ArithUnary2 <- function(func, x) {
    switch(.type(x),
        double=.Map.double(func, x),
        integer=,
        logical=.Map.double(func, as.double(x)),
        .stop("non-numeric argument to mathematical function"))
}

.ArithBinary2 <- function(func, x, y) {
    x <- switch(.type(x),
        double=x,
        integer=,
        logical=,
        NULL=as.double(x),
        .stop("non-numeric argument to mathematical function"))

    y <- switch(.type(y),
        double=y,
        integer=,
        logical=,
        NULL=as.double(y),
        .stop("non-numeric argument to mathematical function"))

    .Map.double(func, x, y)
}

.OrdinalUnary <- function(func, x) {
    switch(.type(x),
        double=.Map.logical(func, x),
        integer=,
        logical=,
        NULL=.Map.logical(func, as.double(x)),
        .stop("non-numeric argument to mathematical function"))
}

.Digits <- function(func, x, y) {
    x <- switch(.type(x),
        double=x,
        integer=,
        logical=,
        NULL=as.double(x),
        .stop("non-numeric argument to mathematical function"))

    y <- switch(.type(y),
        double=,
        integer=,
        logical=,
        NULL=as.integer(y),
        .stop("non-numeric argument to mathematical function"))

    .Map.double(func, x, y)
}

