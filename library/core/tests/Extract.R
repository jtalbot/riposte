
a <- c(10,20,30,40)
a[1L]
a[1.5]
a[c(3L,4L)]
a[c(1,2)]
a[c(TRUE,FALSE)]
a[c(TRUE,FALSE,FALSE,TRUE)]
a[NULL]
a[]
a[vector('integer',0)]
a[-2L]

a[1L] <- 0
(a[1L] <- 5)
a[1L]
a[c(1,2)] <- c(-10,-20)
a


a <- c(a=10,b=20,c=30,d=40)
a[1L]
a[1.5]
a[c(3L,4L)]
a[c(1,2)]
a[c(TRUE,FALSE)]
a[c(TRUE,FALSE,FALSE,TRUE)]
a[NULL]
a[]
a[vector('integer',0)]
a[-2L]

a[1L] <- 0
(a[1L] <- 5)
a[1L]
a[c(1,2)] <- c(-10,-20)
a
a[['b']] <- 100
a
a[['e']] <- 200
a

a[[2L]]
a[[3.4]]

a[[2L]] <- 2.5
a


b <- globalenv()
b[['a']]
b$a
b[['d']] <- 5
b[['d']]
b$k <- 10
b$k

a <- quote(f(1,2,3))
a[[1]]
a[[2]]

a <- list(a=1,b=2,c=c(1,2,3))
a
a[c(1,2)] <- list('a','t')
a
a[[1L]] <- list('j','t')
a

a <- c(1,2,3,4)
dim(a) <- c(2,2)
a[1,1]
a[1,2]
a[2,1]
a[2,2]
a[1,]
a[2,]
a[,1]
a[,2]
a[c(TRUE,FALSE),1]
a[2,c(TRUE,FALSE)]
a[,c(TRUE,FALSE)]
a[c(TRUE,FALSE),]

