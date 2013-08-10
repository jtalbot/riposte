
`~` <- function(y, model) {
    if(missing(model)) {
        r <- list(as.name('~'),
            .pr_expr(.getenv(NULL), 'y')
            )
    }
    else {
        r <- list(as.name('~'),
            .pr_expr(.getenv(NULL), 'y'),
            .pr_expr(.getenv(NULL), 'model')
            )
    }
    attr(r, 'class') <- .characters('formula', 'call')
    attr(r, '.Environment') <- .frame(1L)
    r
}

