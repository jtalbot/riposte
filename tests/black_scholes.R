#adapted from https://github.com/ispc/ispc/tree/master/examples/options
N_OPTIONS <- 65536
N_BLACK_SCHOLES_ROUNDS <- 20
S <- rep(100,1,N_OPTIONS)
X <- rep(98,1,N_OPTIONS)
T <- rep(2, 1,N_OPTIONS)
r <- rep(.02,1,N_OPTIONS)
v <- rep(5,1,N_OPTIONS)

CND <- function(X) {
    L <- abs(X)
    k <- 1.0 / (1.0 + 0.2316419 * L)
    k2 <- k*k
    k3 <- k2*k
    k4 <- k2*k2
    k5 <- k3*k2
    invSqrt2Pi <- 0.39894228040
    w <- (0.31938153 * k - 0.356563782 * k2 + 1.781477937 * k3 + -1.821255978 * k4 + 1.330274429 * k5)
    w <- w * invSqrt2Pi * exp(-L * L * .5)
    #w <- ifelse(X > 0,1 - w,w)
    w <- w + ( (sign(X) + 1.0) / 2.0 ) * (-2.0 * w + 1.0)
    return(w)
}

acc <- 0
for(i in 1:N_BLACK_SCHOLES_ROUNDS) {
	#d1 <- (log10(S/X) + (r + v * v * .5) * T) / (v * sqrt(T))	
	d1 <- (log(S/X) + (r + v * v * .5) * T) / (v * sqrt(T))
	d2 <- d1 - v * sqrt(T)
	result <- S * CND(d1) - X * exp(-r * T) * CND(d2)
	acc <- acc + sum(result)
}
acc <- acc / (N_BLACK_SCHOLES_ROUNDS * N_OPTIONS)
	