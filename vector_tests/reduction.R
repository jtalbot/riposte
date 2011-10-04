
a <- as.double(1:(1024*1024))

system.time(sum(a*a)+prod(a+a))
