
as.character <- function(x, ...) 
    UseMethod('as.character', x)

as.character.default <- function(x, ...)
    as(strip(x), 'character')

is.character <- function(x) 
    UseMethod('is.character', x)

is.character.default <- function(x)
    .type(x) == 'character'

