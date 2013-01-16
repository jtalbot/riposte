
# 4.3.3 Argument evaluation

# actual vs. default evaluation

(f <- function (x, y = x)
{
    x <- x + 1
    y
})
(x <- 1)
f(1, x+2)
f(10)

# call-by-value

(f <- function (x)
{
    x <- x + 1
    x
})
(x <- 100)
f(x)
x

# lazy evaluation

(f <- function (x)
{
    1
})
f(cat("Shouldn't be printed"))

(f(x <- 42))
x

(foo <- function (x)
{
    x
})
(foo(x <- 42))
x

# changes to promise's environment

(x <- 100)
(f <- function (x)
{
    x <<- 1
    x
})
f(x+1)

(x <- 100)
(g <- function (x)
{
    f(x)
})
g(x+1)

#NYI:substitute

# forcing promises
(f <- function (x, y = x + 1)
{
    y
    x <- x + 100
    y
})
f(0)

(f <- function (x, y = x + 1)
{
    x <- x + 100
    y
})
f(0)
