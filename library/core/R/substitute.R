
.substitute <- function(expr, env)
{
    if (is.expression(expr) || is.call(expr)) 
    {
        for(i in seq(1,1,length(expr)))
        {
            expr[[i]] <- .substitute(expr[[i]], env)
        }
    }
    else if (is.symbol(expr) && env != globalenv() && .env_exists(env, expr))
    {
        expr <- .pr_expr(env, expr)
    }

    return(expr)
}

substitute <- function(expr, env=parent.frame(1))
{
    .substitute(.pr_expr(parent.frame(0), quote(expr)), as.environment(env))
}
