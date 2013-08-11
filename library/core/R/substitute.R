
quote <- function(...) {
    r <- strip(`__call__`)
    if(length(r) < 2L)
        Nil
    else if(length(r) > 2L)
        .stop(.pconcat(length(r)-1L, " arguments passed to 'quote' which requires 1"))
    else
        r[[2L]]
}

.substitute <- function(expr, env)
{
    if (is.expression(expr) || is.call(expr)) 
    {
        j <- 1L
        for(i in seq_len(length(expr)))
        {
            if(is.symbol(expr[[i]]) && strip(expr[[i]]) == '...') {
                # expand dots in place
                d <- .get(env, '__dots__')
                n <- .get(env, '__names__')
                if(is.list(.get(env,'__dots__'))) {
                    for(k in seq_len(length(d))) {
                        expr[[j]] <- .pr_expr(d, k)
                        if(is.character.default(.get(env, '__names__')))
                            attr(expr, 'names')[[j]] <- n[[k]]
                        j <- j+1L
                    }
                }
            }
            else {
                expr[[j]] <- .substitute(expr[[i]], env)
                j <- j+1L
            }
        }
    }
    else if (is.symbol(expr) && env != .env_global() && !is.nil(.get(env, expr)))
    {
        expr <- .pr_expr(env, expr)
    }

    return(expr)
}

substitute <- function(expr, env=.frame(1L))
{
    .substitute(.pr_expr(.getenv(NULL), 'expr'), as.environment(env))
}

