
{

library("compiler")
enableJIT(3)

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(as.integer(commandArgs(TRUE)[[1]])/M)

K <- function(x, X) {
    exp(-(x-X)^2)
}

meanshift <- function(x,X) {
    top <- sum(K(x,X)*X)
    bottom <- sum(K(x,X))
    return(top/bottom)
}

X <- runif(M)

run <- function() {
    x <- 0.5
    i <- 1L
    while(i < N) {
        x <- meanshift(x, X)
        i <- i+1L
    }
    x 
}

#cat(run())
cat(system.time(run())[[3]])

}
