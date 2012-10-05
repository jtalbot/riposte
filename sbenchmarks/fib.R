M <- as.integer(commandArgs(TRUE)[[1]])
N <- 100000000 / M

fib <- function(N) {
    a <- runif(M)
    b <- runif(M)
    i <- 0
    while(i < N) {
        i <- i+1
        t <- b
        b <- b+a
        a <- t
    }
    b 
}

system.time(fib(N))
