
attr <- function(x, which, exact=TRUE) {
    if(!.isTRUE(exact))
        .stop('attribute matching is always exact in Riposte')
    attr(x, which)
}

`attr<-` <- function(x, which, value) {
    `attr<-`(x, which, value)
}

