
#m <- c(1,2,3,4,5,6,7,8,9)
#dim(m) <- c(3,3)

N <- 1000

m <- runif(N*N)
dim(m) <- c(N, N)

mv <- function(m, v) {
	r <- 0
	for(i in 1L:ncol(m)) {
		r <- r + m[,i]*v[[i]]	
	}
	r
}

vm <- function(v, m) {
	r <- 0
	for(i in 1L:ncol(m)) {
		r <- r + v[[i]]*m[i,]
	}
	r
}

outer <- function(v) {
	rep(length(v),1L,length(v)^2)
	rep(length(v),length(v),length(v)^2)
}


myqr <- function(m) {
	#q <- diag(ncol(m))
	for(i in 1L:ncol(m)) {
		a <- (m[,i])[i:nrow(m)]
		n <- -sign(m[,i][i])*sqrt(sum(a*a))
		print(n,'\n')
		#v <- ifelse(1:nrow(m) < i, 0,
	#		ifelse(1:nrow(m) == i, m[i,i]-n, m[,i]))
		v <- ifelse(1:nrow(m) < i, 0, m[,i])
		print(v,'\n')
		#v <- matrix(v, ncol=1)
		b <- sum(v*v)
		if(b == 0) next
		m <- m - 2/b * outer((vm(v,m)))
		#m <- m - 2/b * (v %*% (t(v) %*% m))
		#q <- q - 2/b * ((q %*% v) %*% t(v))
	}
	#list(q, m)
	m
}

system.time(myqr(m))
