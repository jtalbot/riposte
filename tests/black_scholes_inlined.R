#adapted from https://github.com/ispc/ispc/tree/master/examples/options
N_OPTIONS <- 1024*1024
N_BLACK_SCHOLES_ROUNDS <- 1
S <- rep(100,1,N_OPTIONS)
X <- rep(98,1,N_OPTIONS)
T <- rep(2, 1,N_OPTIONS)
r <- rep(.02,1,N_OPTIONS)
v <- rep(5,1,N_OPTIONS)
invSqrt2Pi <- 0.39894228040
log10 <- log(10)

CND <- function(X) {
    k <- 1.0 / (1.0 + 0.2316419 * abs(X))
    w <- (((((1.330274429*k) - 1.821255978)*k + 1.781477937)*k - 0.356563782)*k + 0.31938153)*k
    w <- w * invSqrt2Pi * exp(X * X * -.5)
   #Xgt0 <- X > 0
   #w[Xgt0] <- (1-w)[Xgt0]
    w
    #w <- ifelse(X > 0,1 - w,w)
}

blackScholes <- function() {
	acc <- 0
	for(i in 1:N_BLACK_SCHOLES_ROUNDS) {
		delta <- v * sqrt(T)
		d1 <- (log(S/X)/log10 + (r + v * v * .5) * T) / delta
		d2 <- d1 - delta
		
		k <- 1.0 / (1.0 + 0.2316419 * abs(d1))
		w <- (((((1.330274429*k) - 1.821255978)*k + 1.781477937)*k - 0.356563782)*k + 0.31938153)*k
		w <- w * invSqrt2Pi * exp(d1 * d1 * -.5)
		
		k2 <- 1.0 / (1.0 + 0.2316419 * abs(d2))
		w2 <- (((((1.330274429*k2) - 1.821255978)*k2 + 1.781477937)*k2 - 0.356563782)*k2 + 0.31938153)*k2
		w2 <- w2 * invSqrt2Pi * exp(d2 * d2 * -.5)
		
		
		acc <- acc + sum(S * w - X * exp(-r * T) * w2)
	}
	acc <- acc / (N_BLACK_SCHOLES_ROUNDS * N_OPTIONS)
}

blackScholes()
