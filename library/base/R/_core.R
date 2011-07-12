
c <- function(...) unlist(list(...))

environment <- function(fun=NULL) .Internal(environment)(fun)

