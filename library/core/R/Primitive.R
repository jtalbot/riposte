
.Primitive <- function(name)
{
    env <- getRegisteredNamespace('core') 
    env[[strip(name)]]
}

