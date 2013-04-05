
({
    as.environment(1)
})

({
    as.environment(1L)
})

({
    as.environment(parent.frame(1))
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
