
readLines <- function(con, n, ok, warn, encoding) UseMethod('.readLines', con)

.readLines.file <- function(con, n, ok, warn, encoding)
    .External(file_readLines(attr(con, 'conn_id'), n, ok, warn, encoding))


