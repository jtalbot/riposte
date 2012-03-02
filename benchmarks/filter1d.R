
# tests subsetting vector and vector length changing in first iteration of the loop

filter <- function(v, f) {
	r <- 0
	for(i in 1:length(f)) {
		print(r)
		r <- r + v[-c(seq_len(i-1), length(v)-length(f)+seq_len(length(f)-i))]*f[i]
	}
	print(r)
	r
}

a <- as.double(1:(10))
system.time(filter(a, c(-0.5,-1,3,-1,-0.5)))
