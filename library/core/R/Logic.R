
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
    `if`(x, `if`(y, TRUE, FALSE, NA), FALSE, `if`(y, NA, FALSE, NA))
}

`||` <- function(x, y) {
    `if`(x, TRUE, `if`(y, TRUE, FALSE, NA), `if`(y, TRUE, NA, NA))
}

