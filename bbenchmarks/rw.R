
# simulating multiple traces at the same time

rw <- function(n,m) {
    a <- double(m)
    for(i in 1:n) {
        a <- a + ifelse( runif(m), 1, -1 )
    }
}

N <- as.integer(commandArgs(TRUE)[[1]])
M <- as.integer(commandArgs(TRUE)[[2]])

cat(system.time(rw(N/M, M))[[3]])
