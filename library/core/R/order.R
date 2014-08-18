
order <- function(na.last, decreasing, ...) {

    l <- list(...)
   
    if(length(l) == 0L)
        return(vector('integer', 0))
 
    lengths <- .Map(length, list(l), 'double')[[1L]]
    mn <- min(lengths)
    mx <- max(lengths)
    if(mn != mx)
        .stop('argument lengths differ')

    if(mn > 1L) {
        .Riposte('order', .isTRUE(na.last), .isTRUE(decreasing), l)
    }
    else {
        seq_len(mn)
    }
}

