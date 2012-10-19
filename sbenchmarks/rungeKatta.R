
{

library("compiler")
enableJIT(3)

N <- as.integer(commandArgs(TRUE)[[1]])

initial <- function(val) {
	return(sqrt(val))
}
secondary <-function(val,y) {
	return(val*y)
}

rungeKatta <-function(t, h, N) {
	y <- initial(t)
	while (t < N) {
		k1 <- h*secondary(t, initial(t))
		k2 <- h*secondary(t + 0.5*h, initial(t) + 0.5*k1)
		k3 <- h*secondary(t + 0.5*h, initial(t) + 0.5*k2)
		k4 <- h*secondary(t + 0.5*h, initial(t) + k3)
		y <- y + (1.0/6.0)*(k1+k2+k3+k4)
		t <- t +h
	}
	return(y)
}

cat(system.time(rungeKatta(2,1,N))[[3]])

}
