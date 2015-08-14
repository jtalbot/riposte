
assign <- function(x, value, envir, inherits)
{
    print(x)
    if(!inherits || .env_has(envir, x))
    {
        .env_set(envir, x, value)
        return(value)
    }
    else
    {
        while(envir != emptyenv())
        {
            envir <- .getenv(envir)
            if(.env_has(envir, value))
            {
                .env_set(envir, x, value)
                return(value)
            }
        }
    }

    g <- .env_global()
    g[[x]] <- value

    return(value)
}

