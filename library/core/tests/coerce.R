
# NYI: needs access to the search path
({
    #as.environment(1)
})

({
    #as.environment(1L)
})

({
    f <- function() {
        as.environment(parent.frame())
    }
    f()
})

#DIFF: R doesn't support turning an empty list into an environment (?!)
#({
#    as.environment(list())
#})

({
    as.environment(list(x=1))[['x']]
})

({
    as.environment(list(x=1, x=2))[['x']]
})
