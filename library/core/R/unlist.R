
.is.nested <- function(x) {
    as.logical(lapply(x, 
        function(e)
            (is.list(e) 
          && all(attr(e, 'class') != 'expression')
          && all(attr(e, 'class') != 'call'))
         || (is.atomic(e)
          && length(e) != 1) ))
}

.simple.type <- function(x) {
    order <- as.character(list('NULL', 'raw', 'logical', 'integer', 'double', 'character', 'list'))
    types <- as.character(lapply(x, function(e) 
        ifelse(any(attr(e, 'class') == 'name'), 'name', typeof(e))))
    order[[max(match(types, order, 7L, NULL))]]
    
}

.flatten <- function(x, nested, type) {
    if(type == "NULL")
        return(NULL)

    r <- vector(type, 0)
    for(i in seq_len(length(x))) {
        if(nested[[i]]) {
            r[length(r)+seq_len(length(x[[i]]))] <- x[[i]]
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
        n <- rep.int('', length(x))

    r <- vector('character', 0)
    for(i in seq_len(length(x))) {
        if(nested[[i]]) {
            e <- names(x[[i]])
            has.names <- has.names || !is.null(e)
            if(is.null(e))
                e <- rep.int('', length(x[[i]]))
            if(n[[i]] != '')
                e <- .pconcat(n[[i]], 
                    ifelse(e=='', seq_len(length(e)), .pconcat('.',e)))
            r[length(r)+seq_len(length(e))] <- e
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
    if(!is.vector(x, 'any'))
        .stop('argument not a list')

    if(length(x) == 0L)
        return( NULL )

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

            if(length(x) == 0L)
                break

            n <- .is.nested(x)
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

islistfactor <- function(x, recursive) {
    FALSE
}

