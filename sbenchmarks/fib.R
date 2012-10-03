fib <- function(N) {
    a <- 1
    b <- 1
    i <- 0
    while(i < N) {
        i <- i+1
        t <- b
        b <- b+a
        a <- t
    }
    b 
}

system.time(fib(100000000))
