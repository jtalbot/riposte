
readLines <- function(con, n, ok, warn, encoding) 
    .readLines(con, n, ok, warn, encoding)

.readLines <- function(con, n, ok, warn, encoding)
    UseMethod('.readLines', con)

.readLines.file <- function(con, n, ok, warn, encoding) {
    r <- .External('file_readLines', attr(con, 'conn_id'), as.integer(n))
    if(n > 0L && length(r) != n)
        .stop("readLines didn't get enough lines")
    r
}

.readLines.terminal <- function(con, n, ok, warn, encoding) {
    r <- .External('terminal_readLines', strip(con), as.integer(n))
    if(n > 0L && length(r) != n)
        .stop("readLines didn't get enough lines")
    r
}

