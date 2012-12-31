
# test promises and defaults

({
f <- function(x, y) x
z <- 10
f(z, z <- 15)
})

({
f <- function(x, y) {y; x}
z <- 10
f(z, z <- 15)
})

({
f <- function(x, y={x <- 20}) {y; x}
z <- 10
f(z)
})

# tests the fact that you can't simply repass a promise.
# you have to create a new one for each function call, in
# case the value is modified by the called function.
({
g <- function(x) {
    f <- function(z) {x <<- 10; z}
    f(x)
}
g(1)
})

({
f <- function(x=10, y=x) y
f()
})

