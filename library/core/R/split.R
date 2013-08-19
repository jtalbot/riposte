
split <- function(x, f) {
    l <- attr(f, 'levels')
    r <- list()
    for(i in seq_len(length(l))) {
        r[[i]] <- x[strip(f)==i]
    }
    names(r) <- as.character(l)
    r
}

