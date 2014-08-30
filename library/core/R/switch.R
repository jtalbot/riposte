
`switch` <- function(EXPR, ...) {
    EXPR <- strip(EXPR)
    if(.type(EXPR) == 'character') {
        EXPR <- .semijoin(EXPR, `.__names__.`)
        if(EXPR == 0L)
            EXPR <- .semijoin('', `.__names__.`)
    
        while(.env_missing(NULL,EXPR) && EXPR <= ...())
            EXPR <- EXPR+1L
    }

    if(EXPR >= 1 && EXPR <= ...())
        ...(EXPR)
    else
        NULL
}

