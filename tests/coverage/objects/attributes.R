
({
    a <- 1
    attr(a, 'test') <- 'blah'
    attr(a, 'test')
})

({
    attr(a, 'y') <- 10
})

({
    attr(a, 'z')
})

({
    attr(a, 'x') <- c(1,2)
    attr(a, 'x')
})

({
    attr(a, 'x') <- NULL
    attr(a, 'x')
})

#NYI: printing named lists
#NYI: assigning attribute to NULL deletes it.
#({
#    attributes(a)
#})
