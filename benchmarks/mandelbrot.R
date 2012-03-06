#adapted from https://github.com/ispc/ispc/tree/master/examples/mandelbrot

width <- 1024
height <- 768
x0 <- -2
x1 <- 1
y0 <- -1
y1 <- 1


dx <- (x1 - x0) / width
dy <- (y1 - y0) / height
    
c <- (1:(width*height)) - 1
i <- c %% width
j <- floor(c / width)

c_re <- x0 + i * dx
c_im <- y0 + j * dy

mandel <- function(maxIterations) {
	z_re <- c_re
	z_im <- c_im
	cnt <- 0
	for(i in 1:maxIterations) {
		ndone <- z_re * z_re + z_im * z_im <= 4
		z_re <- c_re + ndone * (z_re*z_re - z_im*z_im)
		z_im <- c_im + ndone * (2. * z_re * z_im)
		cnt <- cnt + ndone
	}
	cnt
}

system.time(mandel(100))
