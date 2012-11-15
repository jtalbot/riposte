library(gmp)

line.length <- 10

run <- function(N)
{
	digits <- character(N)
	m <- as.bigz(diag(2))
	
	a <- matrix(c(3,1))
	b <- matrix(c(4,1))
	produce <- matrix(c(1,0,6,3), nrow=2)
	k <- matrix(c(1,0,4,2), nrow=2)
	
	i <- 1
	while(i <= N)
	{
		dr1 <- m %*% a
		d1 <- dr1[[1]] %/% dr1[[2]]
		
		dr2 <- m %*% b
		d2 <- dr2[[1]] %/% dr2[[2]]
		
		if(d1==d2)
		{
			digits[i] <- as.character(d1)
			m <- matrix(c.bigz(10,0,-10*d1,1), ncol=2) %*% m
			i <- i+1
		}
		else
		{
			m <- m %*% produce
			produce <- produce+k	
		}
	}
	lines <- do.call(paste, c(split(digits, (seq_along(digits)-1) %% line.length), sep=""))
	paste(lines, "\t:", seq.int(10,N,by=10), sep="", collapse="\n")
}