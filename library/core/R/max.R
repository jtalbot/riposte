
max <- function(..., na.rm = FALSE) {
    if(...() == 0L) {
        .warning("no non-missing arguments to max; returning -Inf")
        -Inf
    }
    else
        UseGroupMethod('max', 'Summary', ..1)
}

max.default <- function(..., na.rm = FALSE) {
    x <- c(...)

    if(na.rm)
        max(x[!is.na(x)])
    else
        max(x)
}

