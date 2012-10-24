#adapted from https://github.com/ispc/ispc/tree/master/examples/options

{

library("compiler")
enableJIT(3)

N_OPTIONS <- as.integer(commandArgs(TRUE)[[2]])
N_ROUNDS <- as.integer(commandArgs(TRUE)[[1]])/N_OPTIONS

invSqrt2Pi <- 0.39894228040

CND <- function(X) {
    k <- 1.0 / (1.0 + 0.2316419 * abs(X))
    w <- (((((1.330274429*k) - 1.821255978)*k + 1.781477937)*k - 0.356563782)*k + 0.31938153)*k
    w <- w * invSqrt2Pi * exp(X * X * -.5)
    ifelse(X > 0,1 - w,w)
}

black_scholes <- function(S, X, TT, r, v) {
	delta <- v * sqrt(TT)
	d1 <- (log(S/X)/log(10) + (r + v * v * .5) * TT) / delta
	d2 <- d1 - delta
	sum(S * CND(d1) - X * exp(-r * TT) * CND(d2))
}

benchmark <- function() {
    X <- rep(98,each=N_OPTIONS)
    TT <- rep(2,each=N_OPTIONS)
    r <- rep(.02,each=N_OPTIONS)
    v <- rep(5,each=N_OPTIONS)

    S <- 0:(N_OPTIONS-1)

    acc <- 0
    i <- 1L
    while(i <= N_ROUNDS) {
		S <- S+1
        acc <- acc + black_scholes(S, X, TT, r, v)
	    i <- i+1L
    }
}

cat(system.time(benchmark())[[3]])

}
