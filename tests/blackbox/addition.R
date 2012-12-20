(add <- function () 
{
    x <- 1:100
    a <- 0
    for (i in x) a <- a + x[i]
    a
})
add()
