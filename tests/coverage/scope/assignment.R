
({
    a <- 0
    a
})

({
    a = 1
    a
})

({
    2 -> a
    a
})

({
    f <- function(x) {
        a <- 3
    }
    f()
    a
})

({
    f <- function(x) {
        a <<- 4 
    }
    f()
    a
})

({
    f <- function(x) {
        5 ->> a 
    }
    f()
    a
})
