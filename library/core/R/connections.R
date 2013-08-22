
open <- function(con, open, blocking) 
    .open(con, open, blocking)

.open <- function(con, open, blocking)
    UseMethod('.open', con)

close <- function(con, type)
    .close(con, type)

.close <- function(con, type) 
    UseMethod('.close', con)


.connection.seq <- 3L

file <- function(description, open, blocking, encoding, raw) {
    r <- .connection.seq
    .connection.seq <- .connection.seq + 1L
    
    class(r) <- c('file', 'connection')
    attr(r, 'conn_id') <- .External('file_new', description)
    
    if(!identical(open,''))
        .open.file(r, open, blocking)

    r
}

gzfile <- function(description, open, blocking, encoding, raw) {
    r <- .connection.seq
    .connection.seq <- .connection.seq + 1L

    class(r) <- c('gzfile', 'connection')
    attr(r, 'conn_id') <- .External('file_new', description)
    
    if(!identical(open,''))
        .open.file(r, open, blocking)

    r
}

.open.gzfile <- .open.file <- function(con, open, blocking) {
    .External('file_open', attr(con, 'conn_id'), as.character(open))
    NULL
}

.close.gzfile <- .close.file <- function(con, type) {
    .External('file_close', attr(con, 'conn_id'))
    NULL
}

.connection.cat.file <- function(con, x) {
    .External('file_cat', attr(con, 'conn_id'), x)
    NULL
}


.open.terminal <- function(con, open, blocking) NULL
.close.terminal <- function(con, type) NULL
.connection.cat.terminal <- function(con, x) {
    i <- strip(con)
    if(i == 0L) .stop("cannot write to this connection")
    else if(i == 1L) .External('stdout_cat', x)
    else if(i == 2L) .External('stderr_cat', x)
    NULL
}

summary.connection <- function(object) UseMethod('.summary', object)


.summary.gzfile <- .summary.file <- function(con) {
    description <- .External('file_description', attr(con, 'conn_id'))
    r <- list(description, class(con)[[1]], '', '', '', '', '')
    names(r) <- c('description', 'class', 'mode', 'text', 'opened', 'can read', 'can write')
    r
}

