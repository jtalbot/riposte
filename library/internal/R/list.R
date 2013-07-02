
islistfactor <- function(x, recursive) {
    return(is.list(x) && inherits(x, 'factor', FALSE))
}

