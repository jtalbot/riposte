
order <- function(na.last, decreasing, ...) {

    l <- list(...)
   
    if(length(l) == 0L)
        return(vector('integer', 0))
 
    lengths <- as.double(unlist(lapply(l, length),FALSE,FALSE))
    mn <- min(lengths)
    mx <- max(lengths)
    if(mn != mx)
        .stop('argument lengths differ')

    if(mn > 1L) {
        .External('order', .isTRUE(na.last), .isTRUE(decreasing), l)
    }
    else {
        seq_len(mn)
    }
}
