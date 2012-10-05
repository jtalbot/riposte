M <- as.integer(commandArgs(TRUE)[[1]])
N <- 100000000 / M

add <- function(x,y) x+y

fc <- function(V) {
    j <- runif(M)
    i <- 0L
    while(i < N) { i = i+1L; j=add(j,1) }
    j
}

system.time(fc(N))
