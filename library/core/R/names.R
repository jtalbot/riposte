
names <- function(x) UseMethod('names', x)

names.default <- function(x) attr(x, 'names')


`names<-` <- function(x, value) UseMethod('names<-', x)

`names<-.default` <- function(x, value) {
    if(is.nil(value) || is.null(value) || length(value) == 0L)
        attr(x, 'names') <- NULL
    else {
        if(length(value) != length(x))
            .stop("'names' attributes must be the same length as the vector")
        attr(x, 'names') <- as.character(value)
    }
    x
}

