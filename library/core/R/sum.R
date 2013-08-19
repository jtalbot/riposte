
sum <- function(..., na.rm = FALSE) {
    if(...() == 0L)
        0L
    else
        UseGroupMethod('sum', 'Summary', ..1)
}

sum.default <- function(..., na.rm = FALSE) {
    x <- unlist.default(list(...), FALSE, FALSE)

    if(na.rm)
        sum(x[!is.na(x)])
    else
        sum(x)
}

