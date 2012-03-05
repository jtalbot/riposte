
# tests subsetting vector and vector length changing in first iteration of the loop

filter <- function(v, f) {
	r <- 0
	for(i in 1L:length(f)) {
		r <- r + v[(1L:(length(v)-length(f)))+i]*f[i]
	}
	r
}

a <- runif(10000000)
#filter(a, c(0.1,0.15,0.2,0.3,0.2,0.15,0.1))
system.time(filter(a, c(0.1,0.15,0.2,0.3,0.2,0.15,0.1)))
