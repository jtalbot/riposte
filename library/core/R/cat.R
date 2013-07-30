
cat <- function(x, file, sep, fill, labels, append) {
    x <- paste(unlist(x, FALSE, FALSE), sep=sep, collapse=sep)
    if(.isTRUE(fill))
        x <- .pconcat(x, '\n')
    if(identical(file, ''))
        file <- stdout()
    .connection.cat(file, x)
}

.connection.cat <- function(con, x) UseMethod('.connection.cat', con)

