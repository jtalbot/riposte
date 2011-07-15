# The Computer Language Benchmarks Game
# http://shootout.alioth.debian.org
# contributed by Justin Talbot and Saptarshi Guha

mandelbrot <- function(re, im)
{
	C <- complex(real=rep(re, length(im)), imaginary=rep(im, each=length(re)))
	
	Z <- C
	for(i in 1:49)
	{
		Z <- Z^2 + C
		if(all(Mod(Z) >= 2, na.rm=TRUE))
			break
	}
	
	matrix(Mod(Z) >= 2 | is.nan(Re(Z)), nrow=length(im))
}


N <- as.numeric(commandArgs(TRUE)[1])

real <- split(seq(-1.5, 0.5, l=N), (0:(N-1)) %/% 50)
imaginary <- split(seq(-1,1,l=N), (0:(N-1)) %/% 50)

cat("P4\n", N, " ", N, "\n", sep="")
	
result <- logical(0)
for(i in imaginary)
{
	row <- logical(length(i)*N)
    #row <- logical(0)
    j <- 1 
	for(r in real)
	{
        #row <- rbind(row, mandelbrot(r,i))
		row[j:(j+(length(i)*length(r))-1)] <- mandelbrot(r,i)
	}
    print(length(row))
	#result <- cbind(result, row)
}
#result
