
seq.int <- function(...) UseMethod('seq')

seq_along <- function(along.with) seq_len(length(along.with))

seq_len <- function(length.out) seq_len(as.integer(length.out))

