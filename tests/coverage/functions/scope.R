
# 4.3.4 Scope

({
f <- function () 
{
    y <- 10
    g <- function(x) x + y
    return(g) 
}
h <- f()
h(3)
})

