
{

library("compiler")
enableJIT(3)

# simulating multiple traces at the same time

rw <- function(n,m) {
    a <- double(m)
    i <- 1L
    while(i <= n) {
        a <- a + ifelse( runif(m) < 0.5, 1, -1 )
        i <- i+1L
    }
}

N <- as.integer(commandArgs(TRUE)[[1]])
M <- as.integer(commandArgs(TRUE)[[2]])

cat(system.time(rw(N/M, M))[[3]])

}
