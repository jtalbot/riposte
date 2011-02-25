a <- 1
b <- 1
for(i in 1:1000000) {t <- a; a <- b; b <- b+t}
b
