
class <- function(x) {
    r <- attr(x, 'class')
    if(is.null(r)) {
        dim <- attr(x, 'dim')
        if(length(dim) == 2L)
            r <- 'matrix'
        else if(length(dim) > 0L)
            r <- 'array'
        else {
            r <- .type(x)
            if(r == 'double')
                r <- 'numeric'
        }
    }
    r
}

`class<-` <- function(x, value) {
    if(is.null(value) || length(value) == 0L) {
        attr(x, 'class') <- NULL
        x
    }
    else {
        value <- as.character.default(value)
        if(length(value) == 1L) {
            switch(value,
                'logical' = as.logical(x),
                'integer' = as.integer(x),
                'double' = as.double(x),
                'numeric' = as.double(x),
                'complex' = as.complex(x),
                'character' = as.character.default(x),
                'list' = as.list(x),
                {
                    attr(x, 'class') <- as.character.default(value)
                    x
                }
                )
        }
        else {
            attr(x, 'class') <- as.character.default(value)
            x
        }
    }
}

unclass <- function(x) {
    attr(x, 'class') <- NULL
    x
}

inherits <- function(x, what, which) {
    if(which)
        match(what, class(x), 0, NULL)
    else
        any(match(what, class(x), 0, NULL) > 0L)
}

oldClass <- function(x) {
    attr(x, 'class')
}

`oldClass<-` <- function(x, value) {
    if(length(value)==0L)
        attr(x, 'class') <- NULL
    else
        attr(x, 'class') <- as.character.default(value)
    x
}

