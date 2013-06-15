
.find <- function(name, env, type)
{
    while(env != emptyenv())
    {
        if(.env_exists(env, name) && (type == "any" || typeof(env[[name]]) == type))
            return(env[[name]])
        else
            env <- environment(env)
    }
    NULL
}

UseMethod <- function(generic, object, ...)
{
    name <- paste(list(generic, class(object)), ".", NULL)
    fn <- .find(name, parent.frame(1), "closure")
    if(is.null(fn)) 
    {
        name <- paste(list(generic, default), ".", NULL)
        fn <- .find(name, parent.frame(1), "closure")
    }

    if(!is.null(fn))
    {
        fn(object,...)
    }
    else
    {
        stop("UseMethod could not find matching generic")
    }
}
