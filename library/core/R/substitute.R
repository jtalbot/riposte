
quote <- function(expr)
{
    .pr_expr(.getenv(NULL), 'expr')
}

.substitute <- function(expr, env)
{
    if (is.expression(expr) || is.call(expr)) 
    {
        for(i in seq_len(length(expr)))
        {
            expr[[i]] <- .substitute(expr[[i]], env)
        }
    }
    else if (is.symbol(expr) && env != .env_global() && .env_exists(env, expr))
    {
        expr <- .pr_expr(env, expr)
    }

    return(expr)
}

substitute <- function(expr, env=.frame(1L)[[1L]])
{
    .substitute(.pr_expr(.getenv(NULL), 'expr'), as.environment(env))
}
