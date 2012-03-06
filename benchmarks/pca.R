
# pca (via eigen values of covariance matrix) + reprojection onto new basis

N <- 100000L
D <- 50L

a <- read.table("benchmarks/data/pca.txt")
dim(a) <- c(N, D)

cat("done reading\n")

## TODO: this is in its own function so that the intermediate variables go dead
## 	 before this executes. Better liveness handling of registers would fix this
cm <- function(x, y, i, j) {
	cm2(x[(0L:(N-1L))+N*i+1L],y[(0L:(N-1L))+N*j+1L])
}

cov2 <- function(a,b) {
	if(!all(dim(a) == dim(b))) stop("matrices must be same shape")

	m <- nrow(a)
	n <- ncol(a)
	z <- length(a)

	r <- double(n*n)
	for(i in 0L:(n-1L)) {
		for(j in i:(n-1L)) {
			k <- cm(a, b, i, j)
			r[i*n+j+1L] <- k
			r[j*n+i+1L] <- k
		}
	}
	r <- r/(m-1)
	dim(r) <- c(n,n)
	r
}

cov <- function(a,b) {
	if(!all(dim(a) == dim(b))) stop("matrices must be same shape")

	m <- nrow(a)
	n <- ncol(a)
	z <- length(a)

	ma <- double(n)
	mb <- double(n)
	for(i in 1L:n) {
		j <- mean(a[,i])
		k <- mean(b[,i])
		ma[i] <- j
		mb[i] <- k
	}
	
	r <- double(n*n)	
	for(i in 1L:n) {
		for(j in 1L:n) {
			k <- sum((a[,i]-ma[[i]])*(b[,j]-mb[[j]]))
			r[(i-1L)*n+j] <- k
			r[(j-1L)*n+i] <- k
		}
	}
	r <- r/(m-1)
	dim(r) <- c(n,n)
	r
}

## TODO: matrix multiplication cost dominates
## Could just compute the principal components

pca <- function(a) {
	cm <- cov(a,a)
	basis <- eigen(cm, symmetric=TRUE)[[2]]
	#a %*% basis
	basis
}

system.time(pca(a))
#pca(a)
