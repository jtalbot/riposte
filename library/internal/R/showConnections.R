
stdin <- function() {
    x <- 0L
    class(x) <- c('terminal', 'connection')
    x
}

stdout <- function() {
    x <- 1L
    class(x) <- c('terminal', 'connection')
    x
}

stderr <- function() {
    x <- 2L
    class(x) <- c('terminal', 'connection')
    x
}

