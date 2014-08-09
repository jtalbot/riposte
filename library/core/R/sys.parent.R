
sys.call <- function(which) {
    # sys.call has inconsistent behavior on which==0
    # compared to the other sys.* functions!
    if(which <= 0L)
        .frame(as.integer(-which+2L))[['.__call__.']]
    else
        sys.frame(which)[['.__call__.']]
}

sys.frame <- function(which) {
    if(which < 0L)
        .frame(as.integer(-which+2L))
    else {
        n <- 1L
        while(.frame(n) != globalenv()) n <- n+1L
        .frame(as.integer(n-which))
    }
}

sys.nframe <- function() {
    n <- 2L
    while(.frame(n) != globalenv()) { n <- n+1L }
    n-2L
}

sys.function <- function(which) {
    if(which < 0L)
        .frame(as.integer(-which+2L))[['.__function__.']]
    else
        sys.frame(which)[['.__function__.']]
}

sys.parent <- function(n) {
    -n
}

sys.on.exit <- function() {
    .frame(2L)[['.__on.exit__.']]
}

parent.frame <- function(n) {
    .frame(as.integer(n+2L))
}

