f <- function(x,y) x+y
g <- function(...) f(...)
for(i in 1:1000000) g(1,2)
