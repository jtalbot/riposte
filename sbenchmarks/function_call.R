f <- function(x,y) x+y

fc <- function(V) {
    j <- 0
    for(i in V) j <- f(j,1)
    j 
}

V <- 1:10000000
system.time(fc(V))
#fc(10000000)
