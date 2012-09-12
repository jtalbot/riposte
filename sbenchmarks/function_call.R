f <- function(x,y) x+y

fc <- function(N) {
    j <- 0
    for(i in 1:N) j <- f(j,1)
    j
}

system.time(fc(10000000))
