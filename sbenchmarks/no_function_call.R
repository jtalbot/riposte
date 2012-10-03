fc <- function(V) {
    j <- 1.1
    i <- 0L
    while(i < N) { i = i+1L; j=j+1 }
    j
}

N <- 100000000L
system.time(fc(N))
