
{

library("compiler")
enableJIT(3)

# demonstrate lowering ifelse to if

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(commandArgs(TRUE)[[1]]) / M

K <- 10000000L
a <- 1:K

binary.search <- function(v, key) {
    a <- rep(1L, length(key))
    b <- rep(length(v), length(key))

    while(any(a < b)) {
        i <- (a+b) %/% 2L
        a <- ifelse(v[i] < key, i+1L, i)
        b <- ifelse(v[i] < key, b, i)
    }
    return(a)
}

run <- function() {
    for(i in 1L:N) {
        s <- i*(K/N)
        j <- binary.search(a, s:(s+(M-1)))
    }
}

cat(system.time(run())[[3]])

}
