# A simple random walk, since we don't have a random number generator yet, use a constant
# this should really be testing side exits
rw <- function(n) {
    a <- 0
    for(i in 1:n) {
        if(runif(1) < 0.5) a <- a+1
        else a <- a-1
    }
}

N <- as.integer(commandArgs(TRUE)[[1]])

cat(system.time(rw(N))[[3]])
