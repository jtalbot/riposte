
.on.exit <- function(e, expr, add) {
    if(add) {
        # It would be much easier to just build an expression,
        # but this matches R's current behavior
        a <- e[['.__on.exit__.']]
        if(!is.null(a)) {
            if(is.call(a) && identical(strip(a[[1]]), '{')) {
                a[[length(a)+1]] <- expr
                expr <- a
            }
            else {
                expr <- call('{', a, expr)
            }
        }
    }
    e[['.__on.exit__.']] <- expr
    NULL
}

on.exit <- function(expr = NULL, add = FALSE) {
    .on.exit(.frame(1L), .pr_expr(.getenv(NULL), 'expr'), add)
}

