sm <- function(x) { s <- 0; for (y in x) s<-s+y; s }
x <- as.double(1:1000)
system.time(for(i in 1:100000) sm(x))
