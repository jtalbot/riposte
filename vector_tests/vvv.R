is_real_r <- 0
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
		
		if(is_real_r) {
			library("compiler")
			cmp_baseline_fn <- cmpfun(baseline_fn)
			cmp_run_fn <- cmpfun(run_fn)
			
			time_byteline <- system.time(cmp_baseline_fn(N_TIMES,WIDTH)) / N_TIMES
			time_bytecode <- system.time(cmp_run_fn(N_TIMES,WIDTH)) / N_TIMES
			
			time_baseline <- system.time(baseline_fn(N_TIMES,WIDTH)) / N_TIMES
			time_standard <- system.time(run_fn(N_TIMES,WIDTH)) / N_TIMES
			
			report(name,WIDTH,"r_standard",time_standard[3],time_baseline[3])
			report(name,WIDTH,"r_bytecode",time_bytecode[3],time_byteline[3])
		} else {
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






