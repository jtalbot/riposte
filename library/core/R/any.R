
any <- function(..., na.rm = FALSE) {
    if(...() == 0L)
        FALSE
    else
        UseGroupMethod('any', 'Summary', ..1)
}

any.default <- function(..., na.rm = FALSE) {
    x <- c(...)

    if(!is.logical(x) && !is.integer(x))
        .warning(sprintf("coercing argument of type '%s' to logical", .type(x)))

    if(na.rm)
        any(x[!is.na(x)])
    else
        any(x)
}

