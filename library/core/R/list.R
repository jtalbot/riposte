
list <- function(...) list(...)

# .Internal
env2list <- function(x, all.names) {
    names <- .env_names(x)
    r <- list()
    for(n in names) {
        r[[length(r)+1L]] <- x[[n]]
    }
    attr(r, 'names') <- names
    r
}

is.list <- function(x) .type(x) == 'list'

is.pairlist <- function(x) inherits(x, 'pairlist', FALSE)

