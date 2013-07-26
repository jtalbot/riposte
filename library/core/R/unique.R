
unique <- function(x, incomparables, fromLast, nmax) {
    x[!duplicated(x, incomparables, fromLast, nmax)]
}
