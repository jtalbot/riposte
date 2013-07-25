
`levels<-` <- function(x, value) UseMethod('levels<-', x)

`levels<-.default` <- function(x, value) {
    attr(x, 'levels') <- value
    x
}

