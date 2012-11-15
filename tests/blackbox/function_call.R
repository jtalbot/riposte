j <- 0
f <- function(x,y) x+y
for(i in 1:100) j <- f(j,1)
j
