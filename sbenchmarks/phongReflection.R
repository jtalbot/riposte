
N <- as.integer(commandArgs(TRUE)[[1]])

ia <- c(233, 32, 54)
id <- c(32, 34, 64)
is <- c(23, 46, 78)	

normal <- c(1.0, 1.0, 1.0)

phongShading <- function(ks, kd, ka, alpha, lightC, pointC, V) {
	lightM <- lightC - pointC
	mag <- sum(lightM*lightM)
	lightM <- lightM/mag
	rayM <- -lightM

	iP = ka*ia + (kd*(sum(lightM*normal))*id + ks*(sum(rayM*V)^alpha)*is)
	return(iP)
}

lightC <- c(2.4, 5.6, 3.2)
pointC <- c(23.5, 323.5, 434.3)
V <- c(23.4, 553.3, 433.2)

run <- function() {
    total <- 0
    for(i in 1:N) {
        pointC[[1]] <- pointC[[1]]+1
        total <- total + phongShading(2.4, 2.6, 7.3, 4.0, lightC, pointC, V)
    }
}

cat(system.time(run())[[3]])
