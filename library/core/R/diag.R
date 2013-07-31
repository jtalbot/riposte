
diag <- function(x, nrow, ncol) {
    nrow <- as.integer(nrow)
    ncol <- as.integer(ncol)

    m <- rep_len(0, nrow*ncol)
    n <- min(nrow, ncol)
    m[seq_len(n)+(seq_len(n)-1L)*nrow] <- rep_len(x, n)
    attr(m, 'dim') <- c(nrow, ncol)
    m
}

