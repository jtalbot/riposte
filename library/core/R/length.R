
length <- function(x) UseMethod('length', x)

length.default <- function(x) length(strip(x))


`length<-` <- function(x, value) UseMethod('length<-', x)

`length<-.default` <- function(x, value)
{
    if(!is.vector(x))
        .stop("invalid argument")

    a <- NULL

    len <- as.integer(value)
    if(is.na(len))
        .stop("vector size cannot be NA/NaN")

    x[seq_len(len)]
}

