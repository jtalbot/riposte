
(a <- c(10,20,30))
(a[1] <- 40)
a

(a[[2]] <- 50)
a

(a[4] <- 60)
a

(a[c(TRUE,FALSE)] <- c(70, 80))
a

(a[0] <- 0)
a

#NYI: printing vectors with names
#(a <- c(a=10, b=20, c=30))
#(a[['a']] <- 40)
#a

#(a[['d']] <- 60)
#a

#(a[c('a','d')] <- 100)
#a

(a <- list(a, 4))

(a[[1]] <- a)
a

(a[[1]][[1]] <- 200)
a

