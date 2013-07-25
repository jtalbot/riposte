
`!` <- function(x,y) UseGroupMethod('!', 'Ops', x)

`!.default` <- function(x) {
    .LogicCheckUnary(x)
    !strip(x)
}

`&` <- function(x,y) UseMultiMethod('&', 'Ops', x, y)

`&.default` <- function(x, y) {
    .LogicCheckBinary(x,y)
    strip(x) & strip(y)
}

`|` <- function(x,y) UseMultiMethod('|', 'Ops', x, y)

`|.default` <- function(x, y) {
    .LogicCheckBinary(x,y)
    strip(x) | strip(y)
}

`&&` <- function(x, y) {
    .LogicCheckBinary(x,y)
    strip(x) && strip(y)
}

`||` <- function(x, y) {
    .LogicCheckBinary(x,y)
    strip(x) || strip(y)
}

