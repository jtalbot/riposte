

.format.attributes <- function(x, prefix='', show.names=FALSE) {
    # TODO: nested attributes need to use the prefix
    a <- attributes(x)
    if(!is.null(a)) {
        n <- names(a)
        r <- ''
        for(i in seq_len(length(a))) {
            if(show.names || n[[i]] != 'names') {
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
    UseMethod("format", x, ...)
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

format.list <- function(x, ..., prefix='')
{
    if(length(x) == 0)
        'list()'
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
        .pconcat(r, .format.attributes(x, prefix=prefix))
    }
}

format.environment <- function(x, ...)
{
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


deparse <- function(x, ...) UseMethod("deparse", x, ...)

deparse.default <- function(x)
{
    .type(x)
}

deparse.logical <- function(x)
{
    .External(deparse(x))
}

deparse.integer <- function(x)
{
    .External(deparse(x))
}

deparse.double <- function(x)
{
    .External(deparse(x))
}

deparse.character <- function(x)
{
    .External(deparse(x))
}

deparse.list <- function(x)
{
    .External(deparse(x))
}

deparse.environment <- function(x)
{
    .External(deparse(x))
}

deparse.NULL <- function(x)
{
    "NULL"
}

deparse.function <- function(x)
{
    .External(deparse(x))
}

format.name <- deparse.name <- function(x, ...)
{
    strip(x)
}

deparse.indexing <- function(x, open, close) {
    if(length(x) >= 2)
        func <- .pconcat(deparse(x[[2]]), open)
    else
        func <- .pconcat('NULL', open)
    
    if(length(x) >= 3) {
        for(i in 2L+seq_len(length(x)-2L)) {
        
            func <- .pconcat(func, deparse(x[[i]]))
            if(i < length(x))
                func <- .pconcat(func, ', ')
        }
    }
    .pconcat(func, close)
}

format.call <- deparse.call <- function(x, ...)
{
    if(length(x) == 0) {
        "NULL"
    }
    else {
        # TODO: vectorize
        e <- x[[1]]
        func <- format(e)

        if(length(x) == 2 && (func == '+' || func == '-' || func == '!')) {
            .pconcat(func, deparse(x[[2]]))
        }
        else if(length(x) == 3 && (
            func == '/' ||
            func == '%/%' ||
            func == '^' ||
            func == '%%' ||
            func == '$' ||
            func == '@' ||
            func == ':' ||
            func == '::' ||
            func == ':::')) {
            .concat(list(deparse(x[[2]]), func, deparse(x[[3]])))
        }
        else if(length(x) == 3 && (
            func == '+' ||
            func == '-' ||
            func == '*' ||
            func == '~' ||
            func == '&' ||
            func == '|' ||
            func == '&&' ||
            func == '||' ||
            func == '==' ||
            func == '!=' ||
            func == '<' ||
            func == '<=' ||
            func == '>' ||
            func == '>=' ||
            func == '<-' ||
            func == '<<-')) {
            .concat(list(deparse(x[[2]]), func, deparse(x[[3]])))
        }
        else if(func == '[') {
            deparse.indexing(x, '[', ']') 
        }
        else if(func == '[[') {
            deparse.indexing(x, '[[', ']]') 
        }
        else if(func == '{') {
            func <- .pconcat(func,'\n')
            for(i in 1L+seq_len(length(x)-1L)) {
                func <- .pconcat(func, .pconcat('\t',deparse(x[[i]])))
                func <- .pconcat(func,'\n')
            }
            .pconcat(func, '}')
        }
        else {
            func <- .pconcat(func, '(')
            for(i in 1L+seq_len(length(x)-1L)) {
        
                func <- .pconcat(func, deparse(x[[i]]))
                if(i < length(x))
                    func <- .pconcat(func, ', ')
            }
            .pconcat(func, ')')
        }
    }
}

format.expression <- deparse.expression <- function(x, ...)
{
    func <- 'expression(' 
    if(length(x) > 0) {
        for(i in seq_len(length(x))) {
            func <- .pconcat(func, deparse(x[[i]]))
            if(i < length(x))
               func <- .pconcat(func, ', ')
        }
    }
    .pconcat(func, ')')
}


print <- function(x, ...)
{
    text <- format(x)
    cat(text, '\n')
    x
}

