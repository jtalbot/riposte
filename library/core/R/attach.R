
attach <- function(what, pos, name) {
    
    e <- globalenv()
    pos <- as.integer(pos)
    while(pos > 2L) {
        e <- .getenv(e)
        pos <- pos-1L
    }

    r <- .env_new(emptyenv())
    attr(r, 'name') <- name
    if(is.environment(what)) {
        for(f in .env_names(what))
            r[[f]] <- what[[f]]
    }
    else if(is.list(what) && names(what) != NULL) {
        n <- names(what)
        for(i in seq_len(length(what))) {
            r[[n[[i]]]] <- what[[i]]
        }
    }
    
    .setenv(r, .getenv(e))
    .setenv(e, r)

    r 
}

detach <- function(pos) {
    
    e <- globalenv()
    pos <- as.integer(pos)
    while(pos > 2L) {
        e <- .getenv(e)
        pos <- pos-1L
    }

    .setenv(e, .getenv(.getenv(e)))
}
