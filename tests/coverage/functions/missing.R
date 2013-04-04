
({
    f <- function(x,y) missing(x)
    f(1,2)
})

({
    f <- function(x,y) missing(x)
    f(y=2)
})

({
    f <- function(x,y) missing(x)
    f()
})

({
    f <- function(x,y) missing(y)
    f(1)
})

#DIFF: in R missing is recursive. If this function was passed a parameter that
#   was missing in the outer scope, then it should be missing here too. But
#   missingness doesn't propogate through expressions, leading to strange behavior:
#   f <- function(x,y) g(x,y)
#   g <- function(x,y) missing(y)
#   f(1) => TRUE
#   but
#   f <- function(x,y) g(x,y+1)
#   g <- function(x,y) missing(y)
#   f(1) => FALSE
#   but
#   f <- function(x,y) g(x,y+1)
#   g <- function(x,y) y
#   f(1) => Error in y+1: 'y' is missing
#   For now I'll keep the simpler non-recursive semantics. Missing solely means
#   whether or not this scope was passed a value, irregardless of whether that
#   value is missing at a higher level.
#({
#    f <- function(x) missing(x)
#    g <- function(x) f(x)
#    g()
#})
