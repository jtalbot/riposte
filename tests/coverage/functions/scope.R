
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

({
f <- function ()
{
    1+1
}
g <- function ()
{
    f <- 10
    f()
}
g()
})
