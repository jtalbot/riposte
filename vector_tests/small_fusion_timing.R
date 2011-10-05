trace.config(0)
time_many_sizes <- function(name, times, init_fn, run_fn) {
	for(i in 6:23) {
		WIDTH <- 2 ** i
		N_TIMES <- times * (2 ** 23) / WIDTH
		init_fn(WIDTH)
		
		
		trace.config(0)
		time_scalar <- system.time(run_fn(N_TIMES,width)) / N_TIMES
		trace.config(1)
		time_vector <- system.time(run_fn(N_TIMES,width)) / N_TIMES
		trace.config(2)
		time_compiler <- system.time(run_fn(N_TIMES,width)) / N_TIMES
		trace.config(0)
		
		cat(name)
		cat("\t")
		cat(WIDTH)
		cat("\t")
		cat(time_scalar); cat("\t"); cat(time_vector); cat("\t"); cat(time_compiler); cat("\t")
		cat(WIDTH/time_scalar); cat("\t"); cat(WIDTH/time_vector); cat("\t"); cat(WIDTH/time_compiler); cat("\t")
		cat("\n")
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

time_many_sizes("v+v+v",8,init_simple, simple)






