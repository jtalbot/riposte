
.is.nested <- function(x) {
    as.logical(lapply(x, 
        function(e)
            (is.list(e) 
          && all(attr(e, 'class') != 'expression')
          && all(attr(e, 'class') != 'call'))
         || (is.vector(e)
          && length(e) != 1) ))
}

.simple.type <- function(x) {
    order <- as.character(list('NULL', 'raw', 'logical', 'integer', 'double', 'character', 'list'))
    types <- as.character(lapply(x, function(e) 
        ifelse(any(attr(e, 'class') == 'name'), 'name', typeof(e))))
    order[[max(match(types, order, 'list', NULL))]]
}

.flatten <- function(x, nested, type) {
    r <- vector(type, 0)
    for(i in seq(1L, 1L, length(x))) {
        if(nested[[i]]) {
            r[seq((length(r)+1L),1L,length(x[[i]]))] <- x[[i]]
        }
        else {
            r[[length(r)+1L]] <- x[[i]]
        }
    }
    r
}

.flatten.names <- function(x, nested) {
    n <- names(x)
    has.names <- !is.null(n)
    if(is.null(n))
        n <- rep('', length(x))

    r <- vector('character', 0)
    for(i in seq(1L, 1L, length(x))) {
        if(nested[[i]]) {
            e <- names(x[[i]])
            has.names <- has.names || !is.null(e)
            if(is.null(e))
                e <- rep('', length(x[[i]]))
            if(n[[i]] != '')
                e <- .pconcat(n[[i]], 
                    ifelse(e=='', seq(1L, 1L, length(e)), .pconcat('.',e)))
            r[seq((length(r)+1L),1L,length(e))] <- e
        }
        else {
            r[[length(r)+1L]] <- n[[i]]
        }
    }
    if(has.names)
        r
    else
        NULL
}

unlist <- function(x, recursive, use.names) UseMethod('unlist', x)

unlist.default <- function(x, recursive, use.names) {
    if(!is.vector(x))
        stop('argument not a list')

    if(!use.names)
        x <- strip(x)

    if(is.list(x)) {
        n <- .is.nested(x)
        t <- .simple.type(x)
        while(recursive && is.list(x) && (t != 'list' || any(n))) {
            y <- .flatten(x, n, t)
            if(use.names)
                names(y) <- .flatten.names(x, n)
            x <- y
            f <- .is.nested(x)
            t <- .simple.type(x)
        }
        
        if(is.list(x) && (t != 'list' || any(n))) {
            y <- .flatten(x, n, t)
            if(use.names)
                names(y) <- .flatten.names(x, n)
            x <- y
        }
    } 
    x
}
