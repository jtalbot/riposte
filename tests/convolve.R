n <- 400
a <- rep(1,1,n)
b <- rep(1,1,n)
o <- rep(0,1,2*n)

for(i in 1:n) {
	for(j in 1:n) {
		o[i+j] <- o[i+j] + a[i] * b[j]
	}
}
o
