#adapted from https://github.com/ispc/ispc/tree/master/examples/mandelbrot

mandel <- function(c_re, c_im, count) {
	z_re <- c_re
	z_im <- c_im
	cnt <- 0
	for(i in 1:count) {
		ndone <- z_re * z_re + z_im * z_im <= 4.
		z_re <- c_re + ndone * (z_re*z_re - z_im*z_im)
		z_im <- c_im + ndone * (2. * z_re * z_im)
    	cnt <- cnt + ndone
    }
    return(round(cnt/32))
}


width <- 512
height <- 512
x0 <- -2
x1 <- 1
y0 <- -1
y1 <- 1
maxIterations <- 256

dx <- (x1 - x0) / width
dy <- (y1 - y0) / height
    
c <- (1:(width*height)) - 1
i <- c %% width
j <- floor(c / width)

x <- x0 + i * dx
y <- y0 + j * dy
	

#for(j in 1:height) {
#	cat(r[width * (j - 1) + 1:width])
#	cat("\n")
#}

	
trace.config(TRUE,FALSE)
cat(system.time(r <- mandel(x,y,maxIterations)))
cat(" trace on\n")
trace.config(FALSE,FALSE)
cat(system.time(r <- mandel(x,y,maxIterations)))
cat(" trace off\n")

	
	
