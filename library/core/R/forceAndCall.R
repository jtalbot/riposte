
forceAndCall <- function(n, FUN, ...)
{
    for(i in 1:n)
        ...(i)
    FUN(...)
} 
