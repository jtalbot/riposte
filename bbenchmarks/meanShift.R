
{

library("compiler")
enableJIT(3)

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(commandArgs(TRUE)[[1]])/M

K <- function(x, X) {
    exp(-(x-X)^2)
}

meanshift <- function(x,X) {
    top <- sum(K(x,X)*x)
    bottom <- sum(K(x,X))
    return(top/bottom)
}

X <- runif(M)

run <- function() {
    x <- 0
    for(i in 1:N) {
        x <- meanshift(x, X)
    }
    x 
}

cat(system.time(run())[[3]])

}
