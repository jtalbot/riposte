
cat <- function(x, file, sep, fill, labels, append) {
    x <- paste(as.character(x), sep=sep, collapse=sep)
    .cat(file, x)
}

.cat <- function(con, x) UseMethod('.cat')

