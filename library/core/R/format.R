
format <- function(x, trim, digits, nsmall, width, something, na.encode, scientific) {
    .format(x)
}

.format <- function(x, ...) {
    if(is.nil(x))
        ''
    else
        UseMethod(".format", x)
}


.format.attributes <- function(x, prefix='', show.names=FALSE) {
    a <- attributes(x)
    if(!is.null(a)) {
        n <- names(a)
        r <- ''
        for(i in seq_len(length(a))) {
            if((show.names || n[[i]] != 'names')
               && n[[i]] != 'comment') {
                r <- .pconcat(r, '\n')
                p <- .pconcat(prefix, 
                        .pconcat(
                            .pconcat('attr(,"',n[[i]]),'")'))
                r <- .pconcat(.pconcat(r, p), '\n')
                r <- .pconcat(r, .format(a[[i]], prefix=p))
            }
        }
        r
    }
    else {
        ''
    }
}

.format.default <- function(x, ...)
{
    .Riposte('print', x)
}

.format.logical <- function(x, ...)
{
    .Riposte('print', x)
}

.format.integer <- function(x, ...)
{
    .Riposte('print', x)
}

.format.double <- function(x, ...)
{
    .Riposte('print', x)
}

.format.character <- function(x, ...)
{
    #x <- ifelse(is.na(x), "NA", paste(list('"', .escape(x), '"'), "", NULL))
    #.format.vector(x, 'character')
    .Riposte('print', x)
}

.format.pairlist <- .format.list <- function(x, ..., prefix='')
{
    if(length(x) == 0)
        r <- 'list()'
    else {
        r <- ''
        n <- names(x)
        for(i in seq_len(length(x))) {
            if(is.null(n) || identical(n[[i]],'')) {
                p <- .pconcat(.pconcat('[[', i), ']]')
            }
            else {
                p <- .pconcat('$', ifelse(is.na(n[[i]]), '<NA>', n[[i]]))
            }
            r <- .pconcat(r, .pconcat(.pconcat(prefix, p),'\n'))
            r <- .pconcat(r, .format(x[[i]], prefix=p))
            if(i != length(x))
                r <- .pconcat(r,'\n\n')
            else
                r <- .pconcat(r,'\n')
        }
    }
    .pconcat(r, .format.attributes(x, prefix=prefix))
}

.format.environment <- function(x, ...)
{
    if(!is.null(attr(x, 'name')))
        sprintf('<environment: %s>', attr(x, 'name'))
    else
        .Riposte('print', x)
}

.format.NULL <- function(x, ...)
{
    "NULL"
}

.format.function <- function(x, ...)
{
    .Riposte('print', x)
}

.format.call <- function(x, ...)
{
    .deparse.call(x)
}

.format.expression <- function(x, ...)
{
    func <- 'expression(' 
    if(length(x) > 0) {
        for(i in seq_len(length(x))) {
            func <- .pconcat(func, .deparse(x[[i]]))
            if(i < length(x))
               func <- .pconcat(func, ', ')
        }
    }
    .pconcat(func, ')')
}


