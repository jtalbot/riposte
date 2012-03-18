
# pca (via eigen values of covariance matrix) + reprojection onto new basis

N <- 100000L
D <- 50L

a <- read.table("benchmarks/data/pca.txt")[[1]]
dim(a) <- c(N, D)

cat("done reading\n")

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
		ma[[i]] <- j 
		mb[[i]] <- k
	}
	
	r <- double(0)
	#for(i in 1L:n) {
	#	for(j in i:n) {
	#		k <- sum((a[,i]-ma[[i]])*(b[,j]-mb[[j]]))
	#		r[[(i-1L)*n+j]] <- k
	#		r[[(j-1L)*n+i]] <- k
	#	}
	#}
	bs <- 6
	r <- double(2500)
	force(r)
	for(ii in 1L:ceiling(n/bs)) {
		for(jj in 1L:ceiling(n/bs)) {
			for(io in 1L:bs) {
				for(jo in 1L:bs) {
					i <- as.integer( (ii-1)*bs + (io-1) + 1 )
					j <- as.integer( (jj-1)*bs + (jo-1) + 1 )
					if(j >= i && i <= n && j <= n) {
						#cat(i, " " , j, "\n")
						k <- sum((a[,i]-ma[[i]])*(b[,j]-mb[[j]]))
						r[[(i-1L)*n+j]] <- k
						r[[(j-1L)*n+i]] <- k
					}
					#cat(r[[(38-1L)*n + 45]]/(m - 1),"\n")
				}
			}
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
	cm
	#basis <- eigen(cm, symmetric=TRUE)[[2]]
	#basis
}
#system.time(f <- pca(a))
cat(pca(a))
