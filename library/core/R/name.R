
as.symbol <- as.name <- function(x) {
    if(length(x) != 1 || !is.atomic(x))
        stop("invalid type/length in vector allocation")
    x <- as.character(x)
    if(is.na(x))
        x <- 'NA'
    if(identical(x, ''))
        stop('attempt to use zero-length variable name')
    class(x) <- 'name'
    x
}

is.symbol <- is.name <- function(x) {
    any('name' == class(x))
}
