
(f <- function (x) 
parent.frame(x))
f(1)

# DIFF: R treats the global environment as its own parent, Riposte doesn't yet and may never do so.
#f(2)

(g <- function (x) 
{
    f <- function(x) parent.frame(x)
    f(x)
})
g(2)

# DIFF: R treats the global environment as its own parent, Riposte doesn't yet and may never do so.
#g(3)

