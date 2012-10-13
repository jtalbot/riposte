
M <- as.integer(commandArgs(TRUE)[[1]])
N <- 100000000 / M

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

system.time(run())
