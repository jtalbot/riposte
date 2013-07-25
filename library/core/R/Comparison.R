
`<` <- function(x,y) UseMultiMethod('<', 'Ops', x, y)

`<.default` <- function(x, y) {
    .OrdinalCheckBinary(x,y)
    strip(x)<strip(y)
}

`>` <- function(x,y) UseMultiMethod('>', 'Ops', x, y)

`>.default` <- function(x, y) {
    .OrdinalCheckBinary(x,y)
    strip(x)>strip(y)
}

`<=` <- function(x,y) UseMultiMethod('<=', 'Ops', x, y)

`<=.default` <- function(x, y) {
    .OrdinalCheckBinary(x,y)
    strip(x)<=strip(y)
}

`>=` <- function(x,y) UseMultiMethod('>=', 'Ops', x, y)

`>=.default` <- function(x, y) {
    .OrdinalCheckBinary(x,y)
    strip(x)>=strip(y)
}

`==` <- function(x,y) UseMultiMethod('==', 'Ops', x, y)

`==.default` <- function(x, y) {
    .OrdinalCheckBinary(x,y)
    strip(x)==strip(y)
}

`==.environment` <- function(x, y) {
    strip(x)==strip(y)
}

`!=` <- function(x,y) UseMultiMethod('!=', 'Ops', x, y)

`!=.default` <- function(x, y) {
    .OrdinalCheckBinary(x,y)
    strip(x)!=strip(y)
}

