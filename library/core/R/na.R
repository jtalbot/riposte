
is.na <- function(x) UseMethod('is.na', x)

is.na.default <- function(x) is.na(strip(x))

anyNA <- function(x) any(is.na(strip(x)))

