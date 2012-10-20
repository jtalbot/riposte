
{

library("compiler")
enableJIT(3)

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(as.integer(commandArgs(TRUE)[[1]]) / M)

fib <- function(N, M) {
    a <- rep(0,M)
    b <- rep(1,M)
    i <- 0L
    while(i < N) {
        i <- i+1L
        t <- b
        b <- b+a
        a <- t
    }
    b 
}

cat(system.time(fib(N, M))[[3]])

}
