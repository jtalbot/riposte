
invisible <- function(x=NULL) .invisible(x)

withVisible <- function(x) {
    r <- .withVisible(x)
    names(r) <- .characters('value', 'visible')
    r
}
