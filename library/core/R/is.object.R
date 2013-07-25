
is.object <- function(x) {
    !is.null(attr(x, 'class'))
}

