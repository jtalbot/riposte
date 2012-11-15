# The Computer Language Benchmarks Game
# http://shootout.alioth.debian.org/
# contributed by Justin Talbot

solar.mass <- 4*pi*pi
days.per.year <- 365.24

p <- cbind(
	c( 0,
         4.84143144246472090e+00,
         8.34336671824457987e+00,
         1.28943695621391310e+01,
         1.53796971148509165e+01),
	c( 0,
        -1.16032004402742839e+00,
         4.12479856412430479e+00,
        -1.51111514016986312e+01,
        -2.59193146099879641e+01),
	c( 0,
        -1.03622044471123109e-01,
        -4.03523417114321381e-01,
        -2.23307578892655734e-01,
         1.79258772950371181e-01))

v <- cbind(
	c( 0,
         1.66007664274403694e-03,
        -2.76742510726862411e-03,
         2.96460137564761618e-03,
         2.68067772490389322e-03),
	c( 0,
         7.69901118419740425e-03,
         4.99852801234917238e-03,
         2.37847173959480950e-03,
         1.62824170038242295e-03),
	c( 0,
        -6.90460016972063023e-05,
         2.30417297573763929e-05,
        -2.96589568540237556e-05,
        -9.51592254519715870e-05)) * days.per.year

mass <- c( 1,
           9.54791938424326609e-04,
           2.85885980666130812e-04,
           4.36624404335156298e-05,
           5.15138902046611451e-05) * solar.mass

# offset momentum
v[1,] <- -apply(apply(v,2,`*`,mass),2,sum) / solar.mass

pairs <- cbind(c(1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5),
               c(2,3,4,5,1,3,4,5,1,2,4,5,1,2,3,5,1,2,3,4))
#pairs <- cbind(c(1,1,1,1,2,2,2,3,3,4), c(2,3,4,5,3,4,5,4,5,5))
m <- mass[pairs[,2]]

advance <- function(dt) {
    d <- p[pairs[,2],] - p[pairs[,1],]
    a <- d * (apply(d^2, 1, sum)^-1.5 * dt * m)
    
     v[1,] <- v[1,] + apply(a[1:4,], 2, sum)
     v[2,] <- v[2,] + apply(a[5:8,], 2, sum)
     v[3,] <- v[3,] + apply(a[9:12,], 2, sum)
     v[4,] <- v[4,] + apply(a[13:16,], 2, sum)
     v[5,] <- v[5,] + apply(a[17:20,], 2, sum)
}

energy <- function() {
	e <- sum(0.5 * mass * apply(v^2,1,sum))
	for(i in 1:4)
	{
		for(j in (i+1):5)
		{
			d <- sum((p[i,] - p[j,])^2)^0.5
			e <- e - mass[i] * mass[j] / d
		}
	}
	e
}

cat(sprintf("%0.9f\n", energy()))

dt <- 0.01

N <- as.numeric(commandArgs(TRUE)[1])

for(i in 1:N) {
	advance(dt)
    p <- p + v * dt
}
cat(sprintf("%0.9f\n", energy()))

