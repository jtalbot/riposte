
{

library("compiler")
enableJIT(3)

N <- as.integer(commandArgs(TRUE)[[1]])

zeta <- function(N) {
    t = 0.0
    for (j in 1:N) {
        t = 0.0
        for (k in 1:10000) {
            t = t + 1.0/(k*k)
        }
    }
    return(t)
}

cat(system.time(zeta(N))[[3]])

}
