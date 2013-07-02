
strptime <- function(x, format, tz) {
    r <- .Map('strptime_map', list(x,format,tz), c('double', rep('integer', 8)))
    names(r) <-
        c('sec', 'min', 'hour', 'mday', 'mon', 'year', 'wday', 'yday', 'isdst')
    class(r) <- c('POSIXlt', 'POSIXt')
    r
}

# assumes that x is a POSIXlt value
as.POSIXct <- function(x, tz) {
    r <- .Map('mktime_map', x, 'double')[[1]]
    class(r) <- c('POSIXct', 'POSIXt')
    r
}
