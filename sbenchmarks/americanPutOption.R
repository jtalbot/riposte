
americanPut <- function(T, S, K, r, sigma, q, n, p) {
	deltaT <- T/n
	up <- exp((sigma * sqrt(deltaT)))

	p0 <- (up * exp(-r * deltaT) - exp(-q*deltaT)) * up / (up^2 - 1)
	p1 <- exp(-r * deltaT) - p0

	for (i in 1:n) {
		p[i] <- K - S * up ^(2*i - n)
		if (p[i] < 0) {
			p[i] = 0
		}	
	}

	for (j in n-1:1) {
		for (i in 1:j) {
			p[i] <- p0 * p[i] + p1 * p[i+1]
			exercise <- K- S * up^(2*i - j)
			if (exercise > p[i]) {
				p[i] <- exercise
			}
		}
	}
    p
}
input <- 1:10000
system.time(americanPut(100, 50.0, 80.0, 3, 2.0, 3.0, length(input), input))
