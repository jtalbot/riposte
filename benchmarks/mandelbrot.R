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

force(c_re)
force(c_im)

mandel <- function(maxIterations) {
	z_re <- c_re
	z_im <- c_im
	cnt <- 0
	for(i in 1:maxIterations) {
		cnt <- cnt + ifelse(z_re * z_re + z_im * z_im <= 4, 1, 0)
		z_re2 <- c_re + (z_re*z_re - z_im*z_im)
		z_im2 <- c_im + (2. * z_re * z_im)
		z_re <- z_re2
		z_im <- z_im2
	}
	cnt
}

system.time(mandel(100))
