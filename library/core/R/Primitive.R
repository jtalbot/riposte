
.Primitive <- function(name) {
    env <- getRegisteredNamespace('core') 
    .get(env, strip(name))
}

