
sys.call <- function(which) {
    .frame(as.integer(-which+2L))[[2L]]
}

sys.frame <- function(which) {
    .frame(as.integer(-which+2L))[[1L]]
}

sys.nframe <- function() {
    0
}

sys.function <- function(which) {
    .frame(as.integer(-which+2L))[[3L]]
}

sys.nargs <- function(which) {
    .frame(as.integer(-which+2L))[[4L]]
}

sys.parent <- function(n) {
    -n
}

parent.frame <- function(n) {
    .frame(as.integer(n+2L))[[1L]]
}
