
on.exit <- function(expr = NULL, add = FALSE) {
    e <- core::parent.frame()
    expr <- .pr_expr(core::parent.frame(0), 'expr')
    if(add) {
        # It would be much easier to just build an expression,
        # but this matches R's current behavior
        a <- e[['__on.exit__']]
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
    e[['__on.exit__']] <- expr
    NULL
}

sys.on.exit <- function() {
    core::parent.frame()[['__on.exit__']]
}
