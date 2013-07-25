
strptime <- function(x, format, tz) {
    r <- .Map('strptime_map', list(x,format,tz), c('double', rep('integer', 8)))
    names(r) <-
        c('sec', 'min', 'hour', 'mday', 'mon', 'year', 'wday', 'yday', 'isdst')
    attr(r, 'class') <- c('POSIXlt', 'POSIXt')
    r
}

