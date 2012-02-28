
m <- c(1,2,3,4,5,6,7,8,9)
dim(m) <- c(3,3)

myqr <- function(m) {
	q <- diag(ncol(m))
for(i in 1:ncol(m)) {
	a <- m[i:nrow(m),i]
	n <- -sign(m[i,i])*sqrt(sum(a*a))
	v <- ifelse(1:nrow(m) < i, 0,
		ifelse(1:nrow(m) == i, m[i,i]-n, m[,i]))
	v <- matrix(v, ncol=1)
	b <- as.vector(t(v) %*% v)
	if(b == 0) next
	m <- m - 2/b * (v %*% (t(v) %*% m))
	q <- q - 2/b * ((q %*% v) %*% t(v))
}
list(q, m)
}
