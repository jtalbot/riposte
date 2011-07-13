
mandel <- function(c_re, c_im, count) {
	z_re <- c_re
	z_im <- c_im;
	cnt <- 0
	for(i in 1:count) {
        new_re <- z_re*z_re - z_im*z_im
        new_im <- 2. * z_re * z_im
        done <- z_re * z_re + z_im * z_im > 4.
        z_re <- c_re + ifelse(done,0,new_re)
        z_im <- c_im + ifelse(done,0,new_im)
    	cnt <- cnt + ifelse(done,0,1)
    }
    
    return(round(cnt/32))
}


width <- 32
height <- 102
x0 <- -2
x1 <- 1
y0 <- -1
y1 <- 1
maxIterations <- 256

dx <- (x1 - x0) / width
dy <- (y1 - y0) / height
    
for(j in 1:height) {
	i <- 1:width
	x <- x0 + i * dx
	y <- y0 + j * dy
	r <- mandel(x,y,maxIterations)
	#print(r)
}
