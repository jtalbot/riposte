
`~` <- function(y, model) {
    r <- list(as.name('~'),
        .pr_expr(environment(NULL), 'y'),
        .pr_expr(environment(NULL), 'model')
        )
    attr(r, 'class') <- characters('formula', 'call')
    attr(r, '.Environment') <- parent.frame(1L)
    r
}

