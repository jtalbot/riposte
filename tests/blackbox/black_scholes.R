#adapted from https://github.com/ispc/ispc/tree/master/examples/options
({

N_OPTIONS <- 1 
N_BLACK_SCHOLES_ROUNDS <- 1 
S <- rep(100,1,N_OPTIONS)
X <- rep(98,1,N_OPTIONS)
T <- rep(2, 1,N_OPTIONS)
r <- rep(.02,1,N_OPTIONS)
v <- rep(5,1,N_OPTIONS)
invSqrt2Pi <- 0.39894228040

CND <- function(X) {
    k <- 1.0 / (1.0 + 0.2316419 * abs(X))
    w <- (((((1.330274429*k) - 1.821255978)*k + 1.781477937)*k - 0.356563782)*k + 0.31938153)*k
    w <- w * invSqrt2Pi * exp(X * X * -.5)
    #NYI: ifelse
    #w <- ifelse(X > 0,1 - w,w)
}

blackScholes <- function() {
	acc <- 0
	for(i in 1:N_BLACK_SCHOLES_ROUNDS) {
		delta <- v * sqrt(T)
		d1 <- (log(S/X)/log(10) + (r + v * v * .5) * T) / delta
		d2 <- d1 - delta
		acc <- acc + sum(S * CND(d1) - X * exp(-r * T) * CND(d2))
	}
	acc <- acc / (N_BLACK_SCHOLES_ROUNDS * N_OPTIONS)
}

blackScholes()

})
