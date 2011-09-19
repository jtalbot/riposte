
attr <- function(x, which, exact=FALSE) .Internal(attr(x, which, exact))
`attr<-` <- function(x, which, value) .Internal(`attr<-`(x, which, value))

