N <- as.integer(commandArgs(TRUE)[[1]])

fib <- function(N) {
    a <- 0
    b <- 1
    i <- 0L
    while(i < N) {
        i <- i+1L
        t <- b
        b <- b+a
        a <- t
    }
    b 
}

cat(system.time(fib(N))[[3]])
