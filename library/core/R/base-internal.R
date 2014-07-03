
.subset <- function(x, ...) `[`(unclass(x), ...)

.subset2 <- function(x, ...) `[[`(unclass(x), ...)

.isMethodsDispatchOn <- function(onOff) {
    FALSE
}

.seek.file <- function(con, offset) {
    .open.file(con, 'r')
    r <- .External('file_seek', attr(con, 'conn_id'), as.double(offset))
}

lazyLoadDBfetch <- function(key, file, compressed, hook) {

    f <- file(file, 'r', TRUE, '', FALSE)
    .seek.file(f, key[[1]])
    size <- readBin(f, 'int', 1, 4, FALSE, TRUE)
    comp <- '1'
    if(compressed == 2 || compressed == 3)
        comp <- readBin(f, 'character', 1, 1, FALSE, FALSE)
    r <- .readBin.file(f, key[[2]]-5)
    .close.file(f)

    if(compressed)
        r <- .External('decompress', r, size, comp)

    con <- rawConnection('', r, 'r')
    unserializeFromConn(con, NULL)
}
