n <- 10000000

xo <- 0
yo <- 0
zo <- 0
xd <- 1
yd <- 0
zd <- 0

xc <- as.numeric(1:n)
yc <- as.numeric(1:n)
zc <- as.numeric(1:n)

a <- 1
b <- 2*(xd*(xo-xc)+yd*(yo-yc)+zd*(zo-zc))
c <- (xo-xc)*(xo-xc)+(yo-yc)*(yo-yc)+(zo-zc)*(zo-zc)

t0 <- (-b - sqrt(b*b-4*c))/2
t1 <- (-b + sqrt(b*b-4*c))/2

min(min(t0), min(t1))
