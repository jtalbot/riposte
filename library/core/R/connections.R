
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
    attr(r, 'conn_id') <- .Riposte('file_new', description)
    
    if(!identical(open,''))
        .open.file(r, open, blocking)

    r
}

gzfile <- function(description, open, blocking, encoding, raw) {
    r <- .connection.seq
    .connection.seq <- .connection.seq + 1L

    class(r) <- c('gzfile', 'connection')
    attr(r, 'conn_id') <- .Riposte('file_new', description)
    
    if(!identical(open,''))
        .open.file(r, open, blocking)

    r
}

.open.gzfile <- .open.file <- function(con, open, blocking) {
    .Riposte('file_open', attr(con, 'conn_id'), as.character(open))
    NULL
}

.close.gzfile <- .close.file <- function(con, type) {
    .Riposte('file_close', attr(con, 'conn_id'))
    NULL
}

.connection.cat.file <- function(con, x) {
    .Riposte('file_cat', attr(con, 'conn_id'), x)
    NULL
}


.open.terminal <- function(con, open, blocking) NULL
.close.terminal <- function(con, type) NULL
.connection.cat.terminal <- function(con, x) {
    i <- strip(con)
    if(i == 0L) .stop("cannot write to this connection")
    else if(i == 1L) .Riposte('stdout_cat', x)
    else if(i == 2L) .Riposte('stderr_cat', x)
    NULL
}

summary.connection <- function(object) UseMethod('.summary', object)


.summary.gzfile <- .summary.file <- function(con) {
    description <- .Riposte('file_description', attr(con, 'conn_id'))
    r <- list(description, class(con)[[1]], '', '', '', '', '')
    names(r) <- c('description', 'class', 'mode', 'text', 'opened', 'can read', 'can write')
    r
}

rawConnection <- function(description, object, open) {
    r <- .connection.seq
    .connection.seq <- .connection.seq + 1L

    class(r) <- c('rawConnection', 'connection')
    e <- .env_new(emptyenv())
    e$raw <- object
    e$offset <- 0
    attr(r, 'conn') <- e

    r
}

.open.rawConnection <- function(con, open, blocking) NULL
.close.rawConnection <- function(con, type) NULL

textConnection <- function(nm, object, open, env, type) {
    r <- .connection.seq
    .connection.seq <- .connection.seq + 1L

    class(r) <- c('textConnection', 'connection')
    e <- .env_new(emptyenv())
    e$nm <- nm
    e$text <- object
    e$env <- env
    e$offset <- 0
    attr(r, 'conn') <- e

    r
}

.open.textConnection <- function(con, open, blocking) NULL
.close.textConnection <- function(con, type) NULL

