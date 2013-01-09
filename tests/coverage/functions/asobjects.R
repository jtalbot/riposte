
(f <- function (x) 
x + 1)
(g <- function (h) 
h(5))

g(f)

(f <- function () 
function(x) x + 1)
f()(9)

