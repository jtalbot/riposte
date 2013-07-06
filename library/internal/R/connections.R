
open <- function(con, open, blocking) UseMethod('.open', con)
close <- function(con, type) UseMethod('.close', con)


.connection.seq <- 3L

file <- function(description, open, blocking, encoding, raw) {
    r <- .connection.seq
    .connection.seq <- .connection.seq + 1L

    class(r) <- c('file', 'connection')
    attr(r, 'conn_id') <- .External(file_new(description))
    r
}

.open.file <- function(con, open, blocking)
    .External(file_open(attr(con, 'conn_id')))

.close.file <- function(con, type)
    .External(file_close(attr(con, 'conn_id')))

.cat.file <- function(con, x) {
    .External(file_cat(attr(con, 'conn_id'), x))
    NULL
}


stdin <- function() {
    r <- 0L
    class(r) <- c('terminal', 'connection')
    r
}

stdout <- function() {
    r <- 1L
    class(r) <- c('terminal', 'connection')
    r
}

stderr <- function() {
    r <- 2L
    class(r) <- c('terminal', 'connection')
    r
}

.open.terminal <- function(con, open, blocking) NULL
.close.terminal <- function(con, type) NULL
.cat.terminal <- function(con, x) {
    i <- strip(con)
    if(i == 0L) stop("cannot write to this connection")
    else if(i == 1L) .External(stdout_cat(x))
    else if(i == 2L) .External(stderr_cat(x))
    NULL
}
