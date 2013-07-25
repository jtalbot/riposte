
range <- function(..., na.rm = FALSE) {
    if(...() == 0L) {
        .warning("no non-missing arguments to range; returning (Inf, -Inf)")
        c(Inf, -Inf)
    }
    else
        UseGroupMethod('range', 'Summary', ..1)
}

range.default <- function(..., na.rm = FALSE) {
    x <- c(...)

    if(na.rm)
        c(min(x[!is.na(x)]), max(x[!is.na(x)]))
    else
        c(min(x), max(x))
}

