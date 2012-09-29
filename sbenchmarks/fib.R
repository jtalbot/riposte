fib <- function(V) {
    a <- 0
    b <- 1
    for(i in V) {
        t <- b
        b <- b+a
        a <- t
    }
    b
}

V <- 1:10000000
system.time(fib(V))
