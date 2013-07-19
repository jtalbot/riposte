
.rev <- function(x) {
    x[length(x)+1L-seq_len(length(x))]
}

duplicated <- function(x, incomparables, fromLast, nmax) {
    if(identical(incomparables, FALSE))
        incomparables <- NULL

    if(identical(fromLast, TRUE))
        x <- .rev(x)
        
    r <- (match(x, x, 0L, NULL) != seq_len(length(x))) &
         (match(x, incomparables, 0L, NULL) == 0)

    if(identical(fromLast, TRUE))
        r <- .rev(r)

    r
}

anyDuplicated <- function(x, incomparables, fromLast) {
    any((match(x, x, 0L, NULL) != seq_len(length(x))) &
        (match(x, incomparables, 0L, NULL) == 0))
}
