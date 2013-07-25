
dim <- function(x) UseMethod('dim', x)

dim.default <- function(x) attr(x, 'dim')


`dim<-` <- function(x, value) UseMethod('dim<-', x)

`dim<-.default` <- function(x, value) {
    attr(x, 'names') <- NULL
    attr(x, 'dimnames') <- NULL

    if(is.null(x)) {
        attr(x, 'dim') <- NULL
    }
    else {
        if(length(value) == 0L)
            .stop("length-0 dimension vector is invalid")
        if(any(is.na(value)))
            .stop("the dims contain missing values")
        value <- as.integer(value)
        if(any(value < 0L))
            .stop("the dims contain negative values")
        if(prod(value) != length(x))
            .stop(sprintf("dims [product %d] do not match the length of object[%d]", prod(value), length(x)))

        attr(x, 'dim') <- value 
    }
    x
}

