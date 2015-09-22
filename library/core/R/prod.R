
prod <- function(..., na.rm = FALSE) {
    if(...() == 0L)
        1
    else
        UseGroupMethod('prod', 'Summary', ..1)
}

prod.default <- function(..., na.rm = FALSE) {
    x <- as.double(c(...))

    if(na.rm)
        prod(strip(x)[!is.na(strip(x))])
    else
        prod(strip(x))
}

