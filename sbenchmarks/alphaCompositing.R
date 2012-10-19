
{

library("compiler")
enableJIT(3)


alphaCompositing <- function(color1, color2) {
	return(color1 + color2 * (1-color1[[4]]))
}


run <- function(N) {

    color1 <- c(0.5,0.05,0.05,0.1)
    color2 <- c(0.1,0.1,0.1,0.5)
    
    a = 0
    while(a<N) {
        a=a+1;
        color2 = alphaCompositing(color1, color2)
    }
    color2
}

N <- as.integer(commandArgs(TRUE)[[1]])

cat(system.time(run(N))[[3]])

}
