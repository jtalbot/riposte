
is.atomic <- function(x) 
    switch(.type(x), 
        logical=,
        integer=,
        double=,
        complex=,
        character=,
        raw=,
        NULL=TRUE,
        FALSE)

is.recursive <- function(x) 
    !(is.atomic(x) || is.symbol(x))

