
`attr<-` <- function(x, which, value) `attr<-`(x, which, value)

`attributes<-` <- function(obj, value) {
    obj <- strip(obj)
    n <- names(value)

    if(is.null(n))
        stop('attributes must be named')

    for(i in seq(1,1,length(value))) {
        attr(obj, n[[i]]) <- value[[i]]
    }

    obj
}
