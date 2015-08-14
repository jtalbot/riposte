
exists <- function(x, envir, mode, inherits)
{
    mode <- .mode(mode)                 # defined in get.R

    if (.env_has(envir, x))
    {
        if (any(match(typeof(envir[[x]]), mode, 0, NULL)) || mode === "any")
            return(TRUE)
    }
    
    if (inherits)
    {
        while(envir != emptyenv())
        {
            envir <- .getenv(envir)
        
            if (.env_has(envir, x))
            {
                if (any(match(typeof(envir[[x]]), mode, 0, NULL)) ||
                    mode === "any")
                    return(TRUE)
            }
        }
    }

    return(FALSE)
}

get0 <- function(x, envir, mode, inherits, ifnotfound)
{
    mode <- .mode(mode)                 # defined in get.R

    if (.env_has(envir,x))
    {
        val <- envir[[x]]
        if (any(match(typeof(val), mode, 0, NULL)) || mode === "any")
            return(val)
    }

    if (inherits)
    {
        while(envir != emptyenv())
        {
            envir <- .getenv(envir)
    
            if (.env_has(envir,x))
            {
                val <- envir[[x]]
                if (any(match(typeof(val), mode, 0, NULL)) ||
                    mode === "any")
                    return(val)
            }
        }
    }

    return(ifnotfound)
}

