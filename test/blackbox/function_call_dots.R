f <- function(x,y) x+y
g <- function(...) f(...)
for(i in 1:100) g(1,2)
