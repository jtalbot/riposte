
`+` <- function(x,y) {
    if(missing(y))
        UseGroupMethod('+', 'Ops', x)
    else
        UseMultiMethod('+', 'Ops', x, y)
}

`+.default` <- function(x, y) {
    if(missing(y)) {
        .ArithCheckUnary(x)
        x
    }
    else {
        .ArithCheckBinary(x,y)
        strip(x)+strip(y)
    }
}

`-` <- function(x,y) {
    if(missing(y))
        UseGroupMethod('-', 'Ops', x)
    else
        UseMultiMethod('-', 'Ops', x, y)
}

`-.default` <- function(x, y) {
    if(missing(y)) {
        .ArithCheckUnary(x)
        -x
    }
    else {
        .ArithCheckBinary(x,y)
        strip(x)-strip(y)
    }
}

`*` <- function(x,y) UseMultiMethod('*', 'Ops', x, y)

`*.default` <- function(x, y) {
    .ArithCheckBinary(x,y)
    strip(x)*strip(y)
}

`/` <- function(x,y) UseMultiMethod('/', 'Ops', x, y)

`/.default` <- function(x, y) {
    .ArithCheckBinary(x,y)
    strip(x)/strip(y)
}

`^` <- function(x,y) UseMultiMethod('^', 'Ops', x, y)

`^.default` <- function(x, y) {
    .ArithCheckBinary(x,y)
    strip(x)^strip(y)
}

`%%` <- function(x,y) UseMultiMethod('%%', 'Ops', x, y)

`%%.default` <- function(x, y) {
    .ArithCheckBinary(x,y)
    strip(x)%%strip(y)
}

`%/%` <- function(x,y) UseMultiMethod('%/%', 'Ops', x, y)

`%/%.default` <- function(x, y) {
    .ArithCheckBinary(x,y)
    strip(x)%/%strip(y)
}

