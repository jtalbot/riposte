j <- 0
f <- function(x,y) x-y
for(i in 1:10000000) j <- f(j,1)
j
