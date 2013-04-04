
{
a <- new.env(FALSE, emptyenv(), 1)
b <- new.env(FALSE, a, 1)

a[['a']] <- 10

f <- function() a

environment(f) <- b

f()
}
