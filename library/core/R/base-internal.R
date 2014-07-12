
.subset <- function(x, ...) `[`(unclass(x), ...)

.subset2 <- function(x, ...) `[[`(unclass(x), ...)

.isMethodsDispatchOn <- function(onOff) {
    FALSE
}

.seek.file <- function(con, offset) {
    .open.file(con, 'r')
    r <- .External('file_seek', attr(con, 'conn_id'), as.double(offset))
}

makeLazy <- function(n, v, expr, env1, env2) {
    for(i in seq_len(length(n))) {
        a <- as.call(
                list(as.name('lazyLoadDBfetch'),
                    v[[i]],
                    as.name('datafile'),
                    as.name('compressed'),
                    as.name('envhook')))
        promise(n[[i]], a, env1, env2)
    } 
}

lazyLoadDBfetch <- function(key, file, compressed, hook) {
    len <- key[[2]]

    f <- file(file, 'r', TRUE, '', FALSE)
    .seek.file(f, key[[1]])
    size <- readBin(f, 'int', 1, 4, FALSE, TRUE)
    len <- len-4L

    comp <- '1'
    if(compressed == 2 || compressed == 3) {
        comp <- readBin(f, 'character', 1, 1, FALSE, FALSE)
        len <- len-1L
    }

    r <- .readBin.file(f, len)
    .close.file(f)

    if(compressed)
        r <- .External('decompress', r, size, comp)

    con <- rawConnection('', r, 'r')
    unserializeFromConn(con, hook)
}

