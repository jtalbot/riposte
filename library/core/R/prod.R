
prod <- function(..., na.rm = FALSE) {
    if(...() == 0L)
        1
    else
        UseGroupMethod('prod', 'Summary', ..1)
}

prod.default <- function(..., na.rm = FALSE) {
    x <- c(...)

    if(na.rm)
        prod(x[!is.na(x)])
    else
        prod(x)
}

