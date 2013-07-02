
({
    quote(1)
})

({
    quote(1+1)
})

({
    quote(x+y)
})

({
    quote({1;2})
})

({
    substitute(x + y, list(x = 1))
})

({
    substitute(1, list(x=2))
})

({
    x <- 5
    substitute(x, list(y=3))
})

({
    substitute(expression(x + y), list(x = 1))
})

({
    s1 <- function(x, y = substitute(x)) { x; y }
    s1(1)
})

({
    s2 <- function(x, y = substitute(x)) { y; x }
    s2(10)
})

