fib <- function(n) {
    a <- 0
    b <- 1
    for(i in 1:n) {
        t <- b
        b <- b+a
        a <- t
    }
    b
}

system.time(fib(1000000))
