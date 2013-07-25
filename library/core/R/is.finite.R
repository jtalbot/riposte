
is.finite <- function(x) UseMethod('is.finite', x)
is.finite.default <- function(x) {
    if(is.double(x))
        (abs(x) != Inf) & (x == x)
    else
        x == x
}

is.infinite <- function(x) UseMethod('is.infinite', x)
is.infinite.default <- function(x) {
    if(is.double(x))
        (abs(x) == Inf) & (x == x)
    else
        x != x
}

is.nan <- function(x) UseMethod('is.nan', x)
is.nan.default <- function(x) {
    x != x
}

