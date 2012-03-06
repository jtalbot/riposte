n <- 10000000

xo <- 0
yo <- 0
zo <- 0
xd <- 1
yd <- 0
zd <- 0

xc <- as.double(0:(n-1))
yc <- as.double(0:(n-1))
zc <- as.double(0:(n-1))

intersect <- function() {
	rx <- xo-xc
	ry <- yo-yc
	rz <- zo-zc

	a <- 1
	b <- 2*(xd*rx+yd*ry+zd*rz)
	c <- rx*rx+ry*ry+rz*rz-1

	disc <- b*b-4*a*c
		
	t0 <- (-b - sqrt(disc))/2
	t1 <- (-b + sqrt(disc))/2

	cond <- disc > 0
	mt0 <- min(t0[cond])
	mt1 <- min(t1[cond])
	cat(mt1)
	pmin(mt0, mt1)
}

system.time(intersect())
