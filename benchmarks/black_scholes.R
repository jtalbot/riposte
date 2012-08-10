#adapted from https://github.com/ispc/ispc/tree/master/examples/options

N_ROUNDS <- 1
N_OPTIONS <- 10000000

invSqrt2Pi <- 0.39894228040
log10 <- log(10)

S <- rep(100,each=N_OPTIONS)
X <- rep(98,each=N_OPTIONS)
TT <- rep(2,each=N_OPTIONS)
r <- rep(.02,each=N_OPTIONS)
v <- rep(5,each=N_OPTIONS)

force(S)
force(X)
force(TT)
force(r)
force(v)

CND <- function(X) {
    k <- 1.0 / (1.0 + 0.2316419 * abs(X))
    w <- (((((1.330274429*k) - 1.821255978)*k + 1.781477937)*k - 0.356563782)*k + 0.31938153)*k
    w <- w * invSqrt2Pi * exp(X * X * -.5)
    w <- ifelse(X > 0,1 - w,w)
    #Xgt0 <- X > 0
    #w[Xgt0] <- (1-w)[Xgt0]
}

black_scholes <- function() {
	delta <- v * sqrt(TT)
	d1 <- (log(S/X)/log10 + (r + v * v * .5) * TT) / delta
	d2 <- d1 - delta
	sum(S * CND(d1) - X * exp(-r * TT) * CND(d2))
}

benchmark <- function() {
	acc <- 0
	for(i in 1:N_ROUNDS) {
		acc <- acc + black_scholes()
	}
	acc <- acc / (N_ROUNDS * N_OPTIONS)
	acc
}

benchmark()
