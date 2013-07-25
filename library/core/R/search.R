
search <- function() {
    sp <- vector('character', 0)
    e <- globalenv()
    while(!is.null(e)) {
        sp[[length(sp)+1L]] <- attr(e, 'name') 
        e <- .getenv(e)
    }
    sp
}

