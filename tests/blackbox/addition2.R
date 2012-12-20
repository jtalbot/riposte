(x <- 1:10)
(a <- 0)
(add <- function () 
{
    for (i in x) a <<- a + x[i]
})
(add())
