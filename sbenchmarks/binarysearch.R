
{

library("compiler")
enableJIT(3)

M <- as.integer(commandArgs(TRUE)[[1]])
N <- 10000000L

a <- 1:N

binary.search <- function(v, key) {
    a <- 1L
    b <- length(v)

    while(a < b) {
        i <- (a+b) %/% 2L
        if(v[[i]] < key)
            a <- i+1L
        else
            b <- i
    }
    return(a)
}

run <- function() {
    for(i in 1L:M) {
        binary.search(a, i*(N/M))
    }
}

cat(system.time(run())[[3]])

}
