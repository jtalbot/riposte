
# maybe just one more argument?

scan <- function(file, what, nmax, sep, dec, quote, skip, nlines, na.strings, flush, fill, strip.white, quiet, blank.lines.skip, multi.line, comment.char, allowEscapes, encoding) {
    a <- readLines(file, ifelse(nlines > 0L, nlines, -1L), TRUE, FALSE, encoding)
   
    if(comment.char != '') {
        a <- a[substr(a,1L,1L) != comment.char]
    }

    if(is.list(what)) {
        if(sep != '') {
            a <- strsplit(a, sep, TRUE, FALSE, FALSE)
        }
        lengths <- as.integer(lapply(a, length))
        if(!all(lengths == length(what)))
            .stop("all rows not the same length in 'scan'")


        r <- list()
        for(i in seq_len(length(what))) {
            r[[i]] <- as(lapply(a, function(x) x[[i]]), .type(what[[i]]))
        }
        names(r) <- names(what)
        r
    }
    else {
        as(a, .type(what))
    }    
}
