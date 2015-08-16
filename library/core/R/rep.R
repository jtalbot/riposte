
rep <- function(x, ...) UseMethod('rep', x)

rep.default <- function(x, times = 1, length.out = NA_integer_, each = 1)
{
    times <- as.integer(times)
    length.out <- as.integer(length.out)
    each <- as.integer(each)

    if(!is.na(each) && .isTRUE(each != 1L))
        x <- rep.int(x, rep.int(each, length(x)))
   
    if(!is.na(times) && is.na(length.out))
        x <- rep.int(x, times)
    else if(!is.na(length.out))
        x <- rep_len(x, length.out)

    x
}

rep.int <- function(x, times)
{
    times <- as.integer(times)
    if(any(times < 0L))
        .stop("invalid 'times' argument")

    if(length(times) == 1) {
        x[index(length(x), 1, length(x)*times)]
    }
    else if(length(times) == length(x)) {
        unlist(.Map(rep_len, list(x, times)), FALSE, FALSE)
    }
    else {
        .stop("invalid 'times' argument")
    }
}

rep_len <- function(x, length.out)
{
    length.out <- as.integer(length.out)

    if(length(length.out) == 0 || length.out[[1]] < 0)
        .stop("invalid 'length.out' value")

    x[index(length(x), 1, length.out)]
}

