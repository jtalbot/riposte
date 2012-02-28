
# pca (via eigen values of covariance matrix) + reprojection onto new basis

N <- 100000

library(MASS)
a <- mvrnorm(N, c(0,0), matrix(c(0.25,0.2,0.25,0.2),2,2))

outer.columns <- function(x,y,f) {
	m <- nrow(x)
	n <- ncol(x)
	z <- length(x)

	a <- x[rep(1:z, times=n)]
	b <- y[rep(1:m, times=n^2)+rep((1:n)-1, each=z)*m]
	array(f(a,b), dim=c(m,n,n))
}

mycov <- function(a,b) {
	# replace with single pass cov computation 
	if(!all(dim(a) == dim(b))) stop("matrices must be same shape")
	s <- colSums(outer.columns(a,b,`*`)) - outer(colSums(a),colSums(b))/nrow(a)
	s/(nrow(a)-1)
}

pca <- function(a) {
	cm <- mycov(a,a)
	# eigen decomposition
	basis <- eigen(cm, symmetric=TRUE)$vectors
	a %*% basis
}
