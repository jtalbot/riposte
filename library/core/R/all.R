
all <- function(..., na.rm = FALSE) {
    if(...() == 0L)
        TRUE
    else
        UseGroupMethod('all', 'Summary', ..1)
}

all.default <- function(..., na.rm = FALSE) {
    x <- c(...)

    if(!is.logical(x) && !is.integer(x)) {
        .warning(sprintf("coercing argument of type '%s' to logical", .type(x)))
    }
    
    x <- as(strip(x), 'logical')

    if(na.rm)
        all(x[!is.na(x)])
    else
        all(x)
}

