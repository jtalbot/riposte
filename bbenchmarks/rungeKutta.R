
{

library("compiler")
enableJIT(3)

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(commandArgs(TRUE)[[1]])

initial <- function(val) {
	return(sqrt(val))
}
secondary <-function(val,y) {
	return(val*y)
}

rungeKatta <-function(i, h, N) {
	y <- initial(i)
    t <- 1
	while (t < N) {
		k1 <- h*secondary(t, y)
		k2 <- h*secondary(t + 0.5*h, y + 0.5*k1)
		k3 <- h*secondary(t + 0.5*h, y + 0.5*k2)
		k4 <- h*secondary(t + 0.5*h, y + k3)
		y <- y + (1.0/6.0)*(k1+k2+k3+k4)
		t <- t + 1
	}
	return(y)
}

cat(system.time(rungeKatta(1:M,4,N/M))[[3]])

}
