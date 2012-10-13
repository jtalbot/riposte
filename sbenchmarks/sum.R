system.time({a=1; while(a<1000000) a=a+1})

f <- function(x) x+1
system.time({a=1; while(a<1000000) a=f(a)})
