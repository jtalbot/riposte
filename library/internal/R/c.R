c <- function(..., recursive = FALSE) {
    if(...() == 0)
        return(NULL)
    UseMethod('c', ..1)
}

c.default <- function(..., recursive = FALSE) {
    unlist(list(...), recursive, TRUE)
}
