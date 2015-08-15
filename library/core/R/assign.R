
assign <- function(x, value, envir, inherits)
{
    if(!inherits || .env_has(envir, x))
    {
        envir[[x]] <- value
        return(value)
    }
    else
    {
        while(envir != emptyenv())
        {
            envir <- .getenv(envir)
            if(.env_has(envir, value))
            {
                envir[[x]] <- value
                return(value)
            }
        }
    }

    g <- .env_global()
    g[[x]] <- value

    return(value)
}

