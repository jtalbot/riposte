
typeof <- function(x) {
    switch(.type(x),
        'character' = 
            ifelse(any(attr(x,'class')=='name'), 'symbol', 'character'),
        'list' =
            ifelse(any(attr(x,'class')=='call'), 'language',
            ifelse(any(attr(x,'class')=='expression'), 'expression', 'list')),
        .type(x)
    )
}

