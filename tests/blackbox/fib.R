(fib <- function (x)
{
	a <- 1
	b <- 1
	for (i in 1:x) {
        t <- a
        a <- b
        b <- b + t
    }
	b
})

fib(10)
