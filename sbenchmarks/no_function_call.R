fc <- function(V) {
    j <- 0
    for(i in V) j <- j+1
    j
}

V <- 1:10000000
#fc(10000000)
system.time(fc(V))
