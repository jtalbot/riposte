
{
library("compiler")
enableJIT(3)

M <- as.integer(commandArgs(TRUE)[[2]])
N <- sqrt(as.integer(commandArgs(TRUE)[[1]])/M)

americanPut <- function(T, S, K, r, sigma, q, n) {
	deltaT <- T/n
	up <- exp((sigma * sqrt(deltaT)))
	p0 <- (up * exp(-r * deltaT) - exp(-q*deltaT)) * up / (up^2 - 1)
	p1 <- exp(-r * deltaT) - p0

    p <- list()

	for (i in 1:n) {
        p[[i]] <- pmax(0, K - S * up^(2*i-n))
	}

	for (j in (n-1):1) {
		for (i in 1:j) {
            p[[i]] <- pmax( K - S * up^(2*i-j),
                         p0 * p[[i]] + p1 * p[[i+1]] )
		}
	}
    p[[1]]
}

#cat(americanPut(100, 1:M, 80.0, 3, 2.0, 3.0, N))
cat(system.time(americanPut(100, 1:M, 80.0, 3, 2.0, 3.0, N))[[3]])

}
