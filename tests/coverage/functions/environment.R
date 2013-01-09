
(f <- function (x) 
parent.frame(x))
f(1)
f(2)

(g <- function (x) 
{
    f <- function(x) parent.frame(x)
    f(x)
})
g(2)
g(3)

