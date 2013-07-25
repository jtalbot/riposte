
# assumes that x is a POSIXlt value
as.POSIXct <- function(x, tz) {
    r <- .Map('mktime_map', x, 'double')[[1]]
    attr(r, 'class') <- c('POSIXct', 'POSIXt')
    r
}

Date2POSIXlt <- function(x) {
    .stop("Date2POSIXlt (NYI)")
}

# assumes that x is a POSIXct value
as.POSIXlt <- function(x) {
    .stop("as.POSIXlt (NYI)")
}
