

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
                r <- .pconcat(r, format(a[[i]], prefix=p))
            }
        }
        r
    }
    else {
        ''
    }
}

format <- function(x, ...) {
    if(is.nil(x))
        ''
    else
        UseMethod("format", x)
}

format.default <- function(x, ...)
{
    .External(print(x))
}

format.logical <- function(x, ...)
{
    .External(print(x))
}

format.integer <- function(x, ...)
{
    .External(print(x))
}

format.double <- function(x, ...)
{
    .External(print(x))
}

format.character <- function(x, ...)
{
    #x <- ifelse(is.na(x), "NA", paste(list('"', .escape(x), '"'), "", NULL))
    #.format.vector(x, 'character')
    .External(print(x))
}

format.pairlist <- format.list <- function(x, ..., prefix='')
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
            r <- .pconcat(r, format(x[[i]], prefix=p))
            if(i != length(x))
                r <- .pconcat(r,'\n\n')
            else
                r <- .pconcat(r,'\n')
        }
    }
    .pconcat(r, .format.attributes(x, prefix=prefix))
}

format.environment <- function(x, ...)
{
    if(!is.null(attr(x, 'name')))
        sprintf('<environment: %s>', attr(x, 'name'))
    else
        .External(print(x))
}

format.NULL <- function(x, ...)
{
    "NULL"
}

format.function <- function(x, ...)
{
    .External(print(x))
}

format.call <- function(x, ...)
{
    .deparse.call(x)
}

format.expression <- function(x, ...)
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


# This is hidden by print in the base package.
# Include here so we can have minimal printing
# functionality with only the core package.
print <- function(x, ...) UseMethod('print', x)

print.default <- function(x, digits, quote, na.print, print.gap, right, max, useSource, noOpt) 
{
    .cat(format(x), '\n')
    .invisible(x)
}

print.function <- function(x, useSource, ...)
{
    .cat(format.function(x), '\n')
    .invisible(x)
}

