
.rev <- function(x) {
    x[length(x)+1L-seq_len(length(x))]
}

duplicated <- function(x, incomparables, fromLast, nmax) {
    if(incomparables === FALSE)
        incomparables <- NULL

    if(fromLast === TRUE)
        x <- .rev(x)

    r <- (.match(x, x) != seq_len(length(x))) &
         (.match(x, incomparables) == 0L)

    if(fromLast === TRUE)
        r <- .rev(r)

    r
}

anyDuplicated <- function(x, incomparables, fromLast) {
    any( (.match(x, x) != seq_len(length(x))) &
         (.match(x, incomparables) == 0L))
}

