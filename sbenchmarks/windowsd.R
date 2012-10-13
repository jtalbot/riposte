N <- 10000

f <- function() {
    w <- N/2
    x <- rnorm(N)
    z <- rep(0,N)
    for(i in 1:N) {
        a <- max(i-w,1)
        b <- min(i+w,N)
        z[[i]] <- mean(x[a:b])
    }
}

system.time(f())
