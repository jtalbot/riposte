
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
    as.character.default(list(...))
}

.nargs <- function() {
    .frame(1L)[['.__nargs__.']]
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

.cat <- function(..., sep="") .Riposte('cat', list(...), sep)

.warning <- function(msg) {
    warning(TRUE, FALSE, FALSE, msg)
}

.stop <- function(message, call=.frame(1L)[['.__call__.']]) {
    e <- list(message=message, call=call)
    attr(e, 'class') <- c('error', 'condition')

    #.signalCondition(e, message, call)
    .dfltStop(message, call)
}

.escape <- function(x) {
    .Map("escape_map", list(as.character.default(x)), 'character')[[1L]]
}

.pconcat <- function(x, y) .pconcat(strip(x), strip(y))

.concat <- function(x) {
    .Fold("concat", list(as.character.default(x)), 'character')[[1L]]
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
        double=.Map(ffunc, list(x), 'double')[[1L]],
        integer=,
        logical=.Map(ifunc, list(as.integer(x)), 'integer')[[1L]],
        .stop("non-numeric argument to mathematical function"))
}

.ArithUnary2 <- function(func, x) {
    switch(.type(x),
        double=.Map(func, list(x), 'double')[[1L]],
        integer=,
        logical=.Map(func, list(as.double(x)), 'double')[[1L]],
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

    .Map(func, list(x, y), 'double')[[1L]]
}

.OrdinalUnary <- function(func, x) {
    switch(.type(x),
        double=.Map(func, list(x), 'logical')[[1L]],
        integer=,
        logical=,
        NULL=.Map(func, list(as.double(x)), 'logical')[[1L]],
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

    .Map(func, list(x, y), 'double')[[1L]]
}

