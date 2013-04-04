
({
    environment(NULL)
})

({
    f <- function(x) x+1
    environment(f)
})

({
    f <- function(x) x+y
    e <- new.env(FALSE, environment(NULL), 0)
    e[['y']] <- 10
    environment(f) <- e
    f(1)
})
