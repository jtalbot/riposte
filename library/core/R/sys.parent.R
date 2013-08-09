
sys.call <- function(which) {
    .frame(as.integer(-which+2L))[['__call__']]
}

sys.frame <- function(which) {
    .frame(as.integer(-which+2L))
}

sys.nframe <- function() {
    0
}

sys.function <- function(which) {
    .frame(as.integer(-which+2L))[['__function__']]
}

sys.parent <- function(n) {
    -n
}

sys.on.exit <- function() {
    .frame(2L)[['__on.exit__']]
}

parent.frame <- function(n) {
    .frame(as.integer(n+2L))
}

