
{

library("compiler")
enableJIT(3)

rw <- function(n) {
    a <- 0
    for(i in 1:n) {
        if(runif(1) < 0.5) a <- a+1
        else a <- a-1
    }
}

N <- as.integer(commandArgs(TRUE)[[1]])

cat(system.time(rw(N))[[3]])

}
