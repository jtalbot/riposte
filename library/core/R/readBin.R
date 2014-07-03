
readBin <- function(con, what, n, size, signed, swap) {
    size <- as.integer(size)
    n <- as.integer(n)
    if(is.na(size)) {
        size <- switch(what,
            numeric=,
            double=8L,
            integer=,
            int=4L,
            logical=4L,
            complex=16L,
            character=1L,
            raw=1L)
    }
    # TODO: null-terminated strings need special support
    if(!is.raw(con))
        con <- .readBin(con, n*size)
    else
        con <- con[seq_len(n*size)]

    if(.isTRUE(swap) && size > 1L) {
        con <- con[(size-seq_len(size)) + (seq_len(n*size)-1L) %/% size * size + 1L]
    }
 
    switch(what,
            numeric=,
            double=
                if(size == 4 || size == 8)
                    .External('rawToDouble', con, size)
                else
                    .stop(sprintf('size %d is unknown on this machine', size)),
            integer=,
            int=
                if(size == 1 || size == 2 || size == 4 || size == 8)
                    .External('rawToInteger', con, size, .isTRUE(signed))
                else
                    .stop(sprintf('size %d is unknown on this machine', size)),
            logical=
                if(size == 1 || size == 2 || size == 4 || size == 8)
                    .External('rawToLogical', con, size)
                else
                    .stop(sprintf('size %d is unknown on this machine', size)),
            complex=.stop('NYI: rawToComplex'),
            character=.External('rawToChar', con),
            raw=con)
}

.readBin <- function(con, n)
    UseMethod('.readBin', con)

.readBin.file <- function(con, n) {
    .open.file(con, 'rb')
    .External('file_readBin', attr(con, 'conn_id'), as.integer(n))
}

.readBin.gzfile <- function(con, n) {
    .open.gzfile(con, 'rb')
    .External('gzfile_readBin', attr(con, 'conn_id'), as.integer(n))
}

.readBin.rawConnection <- function(con, n) {
    e <- attr(con, 'conn')
    r <- e$raw[e$offset+seq_len(as.integer(n))]
    e$offset <- e$offset + as.integer(n)
    r
}

