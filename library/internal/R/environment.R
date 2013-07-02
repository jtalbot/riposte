
environment <- function(fun) {
    if(is.null(fun))
        core::parent.frame(2)
    else
        .getenv(fun)
}

new.env <- function(hash, parent, size) {
    if(!is.environment(parent))
        stop("'enclos' must be an environment")

    .env_new(parent)
}

parent.env <- function(env) {
    environment(env)
}

`parent.env<-` <- function(fun, value) {
    environment(fun) <- value
    fun
}

environmentName <- function(env) {
    # DIFF: this differs from R, where certain environments have a name that
    # is not an attribute (baseenv, emptyenv, globalenv)
    # We'll see if Riposte can avoid implementing those special cases.
    attr(env, 'name')
}

env.profile <- function(env) {
    # Riposte's hash representation is very different from R's.
    # It's not sensible to return what R returns.
    NULL
}

delayedAssign <- function(x, value, eval.env, assign.env) {
    promise(x, value, eval.env, assign.env)
}
