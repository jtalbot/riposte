
length <- function(x) UseMethod('length', x)

length.default <- function(x) length(strip(x))


`length<-` <- function(x, value) UseMethod('length<-', x)

`length<-.default` <- function(x, value)
{
    if(!is.atomic(x) && !is.list(x))
        .stop("invalid argument")

    a <- NULL

    len <- as.integer(value)
    if(is.na(len))
        .stop("vector size cannot be NA/NaN")

    x[seq_len(len)]
}

lengths <- function(x, use.names)
{
    r <- .Map(length, list(x))
    if(use.names)
        names(r) <- names(x)
    r
}

