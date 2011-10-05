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


init_simple <- function(width) {
	v <<- 1 : width
}

simple <- function(times, width) {
	a <- 0
	for(i in 1:times) {
		a <- a + max(v + v + v)
	}
}

baseline <- function(times,width) {
	a <- 0
	for(i in 1:times) {
		a <- a + max(v)
	}
}

time_many_sizes("v+v+v",8,init_simple, simple, baseline)






