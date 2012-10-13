
N <- as.integer(commandArgs(TRUE)[[1]])

pisum <- function(N) {
    t = 0.0
    for (j in 1:N) {
        t = 0.0
        for (k in 1:10000) {
            t = t + 1.0/(k*k)
        }
    }
    return(t)
}

cat(system.time(pisum(N))[[3]])
