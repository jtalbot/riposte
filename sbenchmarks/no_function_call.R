fc <- function(N) {
    j <- 0
    for(i in 1:N) j <- j+1
    j
}

#fc(10000000)
system.time(fc(10000000))
