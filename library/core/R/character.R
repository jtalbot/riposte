
as.character <- function(x, ...) as(strip(x), 'character')

is.character <- function(x) .type(x) == 'character'

