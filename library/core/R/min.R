
min <- function(..., na.rm = FALSE) {
    if(...() == 0L) {
        .warning("no non-missing arguments to min; returning Inf")
        Inf
    }
    else
        UseGroupMethod('min', 'Summary', ..1)
}

min.default <- function(..., na.rm = FALSE) {
    x <- c(...)

    if(na.rm)
        min(x[!is.na(x)])
    else
        min(x)
}

