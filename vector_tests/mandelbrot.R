#adapted from https://github.com/ispc/ispc/tree/master/examples/mandelbrot
is_real_r <- 1
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


init <- function(V_WIDTH) {
	width <- 32
	height <- V_WIDTH / width
	x0 <- -2
	x1 <- 1
	y0 <- -1
	y1 <- 1
	
	
	dx <- (x1 - x0) / width
	dy <- (y1 - y0) / height
    
	c <- (1:(width*height)) - 1
	i <- c %% width
	j <- floor(c / width)
	
	maxIterations <<- 1
	c_re <<- x0 + i * dx
	c_im <<- y0 + j * dy
}

mandel <- function(maxIterations,width) {
	z_re <- c_re
	z_im <- c_im
	cnt <- 0
	for(i in 1:maxIterations) {
		ndone <- as.double(z_re * z_re + z_im * z_im <= 4.)
		z_re <- c_re + ndone * (z_re*z_re - z_im*z_im)
		z_im <- c_im + ndone * (2. * z_re * z_im)
		cnt <- cnt + ndone
	}
}

baseline <- function(times,width) {
	for(i in 1:times) {
		#end the trace
		result <- max(c_re) 
	}
}

time_many_sizes("mandelbrot",1,init,mandel,baseline)

	
	
