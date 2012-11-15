
f <- function(x, y) {
    z <- 0
    x+z
}
f(4,5)

z <- 10
f <- function(x, y) {
    z <- 0
    rm(z)
    x+z
}
f(4,5)
