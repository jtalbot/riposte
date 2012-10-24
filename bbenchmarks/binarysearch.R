
{

library("compiler")
enableJIT(3)

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(as.integer(commandArgs(TRUE)[[1]]) / M)

K <- 10000000L
a <- 1:K

binary.search <- function(v, key) {
    a <- 1L
    b <- K 

    while(any(a < b)) {
        t <- (a+b) %/% 2L
        a <- ifelse(v[t] < key, t+1L, a)
        b <- ifelse(v[t] < key, b, t)
    }
    return(a)
}

run <- function() {
    i <- 1L
    s <- 1:M
    while(i <= N) {
        s <- s + (K%/%N)
        j <- binary.search(a, s)
        i <- i+1L
    }
}

cat(system.time(run())[[3]])

}
