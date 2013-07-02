
formals <- function(fun) {
    fun[['formals']]
}

`formals<-` <- function(fun, envir, value) {
    stop("NYI: formals<-")
}
