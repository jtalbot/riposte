
# default arguments
(f <- function (x) 
x)
(g <- function (x = 1) 
x)

f(1)
g()

# missing
(f <- function (x, y) 
missing(y))
f(1,2)
f(1)
f(y=1)

f(1,cat("Should not be evaluated"))

#...
(f <- function (...) 
1)
f(1,2,3,4,5,6,7,8,9)

(g <- function (x, ...) 
x)
(f <- function (...) 
g(...))
f(1,2,3,4,5,6)

