
.format.vector <- function(x, type)
{
    if(length(x) == 0)
        cat(.External(paste(list(type, '(0)'), "")))
    else
        cat(x)
}

.format.attributes <- function(x, prefix='', show.names=FALSE) {
    # TODO: nested attributes need to use the prefix
    a <- attributes(x)
    if(!is.null(a)) {
        n <- names(a)
        r <- ''
        for(i in seq(1,1,length(a))) {
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
    format(strip(x), ...) 
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
        for(i in seq(1,1,length(x))) {
            if(is.null(n) || n[[i]] == '') {
                p <- .pconcat(.pconcat('[[', i), ']]')
            }
            else {
                p <- .pconcat('$', n[[i]])
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
        for(i in seq(3,1,length(x)-2)) {
        
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
            paste(list(deparse(x[[2]]), func, deparse(x[[3]])), sep="", collapse="")
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
            paste(list(deparse(x[[2]]), func, deparse(x[[3]])), sep=" ", collapse="")
        }
        else if(func == '[') {
            deparse.indexing(x, '[', ']') 
        }
        else if(func == '[[') {
            deparse.indexing(x, '[[', ']]') 
        }
        else if(func == '{') {
            func <- .pconcat(func,'\n')
            for(i in seq(2,1,length(x)-1)) {
                func <- .pconcat(func, .pconcat('\t',deparse(x[[i]])))
                func <- .pconcat(func,'\n')
            }
            .pconcat(func, '}')
        }
        else {
            func <- .pconcat(func, '(')
            for(i in seq(2,1,length(x)-1)) {
        
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
        for(i in seq(1,1,length(x)-1)) {
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

