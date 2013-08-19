
isS4 <- function(object) {
    !is.null(attr(object, 'isS4'))
}

