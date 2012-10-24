
{
library("compiler")
enableJIT(3)

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(sqrt(as.integer(commandArgs(TRUE)[[1]])/M))

americanPut <- function(T, S, K, r, sigma, q, n) {
	deltaT <- T/n
	up <- exp((sigma * sqrt(deltaT)))
	p0 <- (up * exp(-r * deltaT) - exp(-q*deltaT)) * up / (up^2 - 1)
	p1 <- exp(-r * deltaT) - p0

    p <- list()

    i <- 1
    while(i <= n) {
        p[[i]] <- pmax(0, K - S * up^(2*i-n))
	    i <- i+1
    }

    j <- n-1L
    while(j >= 1L) {
        i <- 1L
        while(i <= j) {
            p[[i]] <- pmax( K - S * up^(2*i-j),
                             p0 * p[[i]] + p1 * p[[i+1]] )
		    i <- i+1L
        }
        j <- j-1L
    }
    p[[1]]
}

#cat(americanPut(100, 1:M, 80.0, 3, 2.0, 3.0, N))
cat(system.time(americanPut(100, 1:M, 80.0, 3, 2.0, 3.0, N))[[3]])

}
