
nchar <- function(x, type, allowNA, keepNA) {
    .nchar(strip(x))
}

nzchar <- function(x) {
    .nchar(strip(x)) > 0
}

