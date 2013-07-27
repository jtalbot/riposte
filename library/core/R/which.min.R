
which.min <- function(x) {
    w <- seq_len(length(x))[min(x[!is.nan(x) & !is.na(x)])==x]
    if(length(w) > 0L)
        w[[1L]]
    else
        w
}

which.max <- function(x) {
    w <- seq_len(length(x))[max(x[!is.nan(x) & !is.na(x)])==x]
    if(length(w) > 0L)
        w[[1L]]
    else
        w
}

