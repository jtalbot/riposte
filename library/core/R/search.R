
search <- function() {
    sp <- vector('character', 0)
    e <- globalenv()
    while(e !== emptyenv()) {
        sp[[length(sp)+1L]] <- attr(e, 'name') 
        e <- .getenv(e)
    }
    sp
}

