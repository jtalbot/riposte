
is.na <- function(x) UseMethod('is.na', x)

is.na.default <- function(x) {
    is.na(strip(x))
}

