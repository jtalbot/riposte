
{

library("compiler")
enableJIT(3)

# computes Riemann zeta function for different real-valued values of s

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(commandArgs(TRUE)[[1]]) / M

zeta <- function(p, N) {
    t <- rep(0, length(p))
    for (k in 1:N) {
        t <- t + 1.0/(k^p)
    }
    return(t)
}

cat(system.time(zeta(3:(3+M-1), N))[[3]])

}
