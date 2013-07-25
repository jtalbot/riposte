
cat <- function(x, file, sep, fill, labels, append) {
    x <- paste(as.character(x), sep=sep, collapse=sep)
    if(identical(file, ''))
        file <- stdout()
    .connection.cat(file, x)
}

.connection.cat <- function(con, x) UseMethod('.connection.cat', con)

