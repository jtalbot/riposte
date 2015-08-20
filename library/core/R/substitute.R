
quote <- function(...) {
    # this roundabout approach is necessary to handle quote(...),
    # which should return `...`, but normal evaluation will
    # substitute for the ... and give the wrong results.
    # Maybe should be special.
    r <- strip(`.__call__.`)
    if(length(r) < 2L)
        .stop("0 arguments passed to 'quote' which requires 1")
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
                d <- env[['.__dots__.']]
                n <- env[['.__names__.']]
                if(is.list(d)) {
                    for(k in seq_len(length(d))) {
                        expr[[j]] <- .pr_expr(d, k)
                        if(is.character.default(n))
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
    else if (is.symbol(expr) && env != .env_global() && .env_has(env, expr))
    {
        expr <- .pr_expr(env, expr)
    }

    return(expr)
}

substitute <- function(expr, env=.frame(1L))
{
    .substitute(.pr_expr(.getenv(NULL), 'expr'), as.environment(env))
}

