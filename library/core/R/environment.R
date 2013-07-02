
environment <- function(fun) {
    if(is.null(fun))
        parent.frame(1)
    else
        .getenv(fun)
}

`environment<-` <- function(fun, value) {
    if(!is.environment(value))
        stop("replacement object is not an environment")

    .setenv(fun, value)
}

# is.environment defined in coerce.R

globalenv <- function() {
    getNamespace('global')
}

emptyenv <- function() {
    getNamespace('empty')
}

baseenv <- function() {
    getNamespace('base')
}

