#adapted from https://github.com/ispc/ispc/tree/master/examples/options
trace.config(0)
time_many_sizes <- function(name, times, init_fn, run_fn, baseline_fn) {
	report <- function(name, width, exec_type, time, baseline_time) {
		cat(name)
		cat("\t")
		cat(as.integer(width))
		cat("\t")
		cat(exec_type)
		cat("\t")
		cat(time) 
		cat("\t")
		cat(baseline_time)
		cat("\n")
	}
	for(i in 6:23) {
		WIDTH <- 2 ** i
		N_TIMES <- times * (2 ** 23) / WIDTH
		init_fn(WIDTH)
		
		
		trace.config(0)
		time_baseline <- system.time(baseline_fn(N_TIMES,WIDTH)) / N_TIMES
		time_scalar <- system.time(run_fn(N_TIMES,WIDTH)) / N_TIMES
		trace.config(1)
		time_vector <- system.time(run_fn(N_TIMES,WIDTH)) / N_TIMES
		trace.config(2)
		time_compiler <- system.time(run_fn(N_TIMES,WIDTH)) / N_TIMES
		trace.config(0)
		
		report(name,WIDTH,"scalar_interpreter",time_scalar, time_baseline)
		report(name,WIDTH,"vector_interpreter",time_vector, time_baseline)
		report(name,WIDTH,"vector_compiler",time_compiler, time_baseline)
		
	}	
}

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

blackScholesInit <- function(N_OPTIONS) {
	S <<- rep(100,1,N_OPTIONS)
	X <<- rep(98,1,N_OPTIONS)
	T <<- rep(2, 1,N_OPTIONS)
	r <<- rep(.02,1,N_OPTIONS)
	v <<- rep(5,1,N_OPTIONS)
}

blackScholes <- function(N_BLACK_SCHOLES_ROUNDS,N_OPTIONS) {
	acc <- 0
	for(i in 1:N_BLACK_SCHOLES_ROUNDS) {
		delta <- v * sqrt(T)
		d1 <- (log(S/X)/log10 + (r + v * v * .5) * T) / delta
		d2 <- d1 - delta
		acc <- acc + sum(S * CND(d1) - X * exp(-r * T) * CND(d2))
	}
	acc <- acc / (N_BLACK_SCHOLES_ROUNDS * N_OPTIONS)
}

baseline <- function(times,width) {
	acc <- 0
	for(i in 1:times) {
		acc <- acc + 1
	}
}

time_many_sizes("black_scholes",1,blackScholesInit,blackScholes, baseline)
