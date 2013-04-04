
({
    f <- function(...) ..1
    f(1,2,3)
})

({
    f <- function(...) ..2
    f(1,2,3)
})

({
    f <- function(...) list(...)
    f(1,2,3)
})
