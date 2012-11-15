
# tests subsetting vector and vector length changing in first iteration of the loop

filter <- function(v, f) {
	r <- 0
	for(i in 1:length(f)) {
		r <- r + v[-c(seq_len(i-1), length(v)-length(f)+seq_len(length(f)-i))]*f[i]
	}
	r
}

a <- as.double(1:(1024*1024))
system.time(filter(a, c(-0.5,-1,3,-1,-0.5)))
