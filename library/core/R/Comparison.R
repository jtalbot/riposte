
`<` <- function(e1,e2) UseMultiMethod('<', 'Ops', e1, e2)

`<.default` <- function(e1, e2) {
    .OrdinalCheckBinary(e1,e2)
    strip(e1)<strip(e2)
}

`>` <- function(e1,e2) UseMultiMethod('>', 'Ops', e1, e2)

`>.default` <- function(e1, e2) {
    .OrdinalCheckBinary(e1,e2)
    strip(e1)>strip(e2)
}

`<=` <- function(e1,e2) UseMultiMethod('<=', 'Ops', e1, e2)

`<=.default` <- function(e1, e2) {
    .OrdinalCheckBinary(e1,e2)
    strip(e1)<=strip(e2)
}

`>=` <- function(e1,e2) UseMultiMethod('>=', 'Ops', e1, e2)

`>=.default` <- function(e1, e2) {
    .OrdinalCheckBinary(e1,e2)
    strip(e1)>=strip(e2)
}

`==` <- function(e1,e2) UseMultiMethod('==', 'Ops', e1, e2)

`==.default` <- function(e1, e2) {
    .OrdinalCheckBinary(e1,e2)
    strip(e1)==strip(e2)
}

`==.environment` <- function(e1, e2) {
    strip(e1)==strip(e2)
}

`!=` <- function(e1,e2) UseMultiMethod('!=', 'Ops', e1, e2)

`!=.default` <- function(e1, e2) {
    .OrdinalCheckBinary(e1,e2)
    strip(e1)!=strip(e2)
}

