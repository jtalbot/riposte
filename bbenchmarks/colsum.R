
{

library("compiler")
enableJIT(3)

# computes Riemann zeta function for different real-valued values of s

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(commandArgs(TRUE)[[1]]) / M

colsum <- function(m) {
    cols <- dim(m)[[2]]

    s <- 0
    for(i in 1:cols) 
    {
        s <- s + m[,i]
    }
    return(s)
}

m <- matrix(runif(M*N), M, N)

cat(system.time(colsum(m))[[3]])

}
