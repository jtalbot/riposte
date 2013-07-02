
rep.int <- function(x, times) {
    if(length(times) == 1) {
        x[index(length(x), 1, length(x)*times)]
    }
    else if(length(times) == length(x)) {
        unlist(mapply(function(x,t) rep(x,t), x, times), FALSE, FALSE)
    }
}

rep_len <- function(x, length.out) {
    length.out <- as.integer(length.out)

    if(length(length.out) == 0 || length.out[[1]] < 0)
        stop("invalid 'length.out' value")

    x[index(length(x), 1, length.out)]
}
