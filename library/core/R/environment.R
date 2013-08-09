
environment <- function(fun) {
    if(is.null(fun))
        .frame(2L)
    else
        .getenv(fun)
}

`environment<-` <- function(fun, value) {
    .setenv(fun, value)
}


is.environment <- function(x) .type(x) == 'environment'

.GlobalEnv <- .env_global()
globalenv <- function() .env_global()

emptyenv <- function() getRegisteredNamespace('empty')
baseenv <- function() getRegisteredNamespace('base')

new.env <- function(hash, parent, size) {
    if(!is.environment(parent))
        .stop("'enclos' must be an environment")

    .env_new(parent)
}


parent.env <- function(env) {
    .getenv(env)
}

`parent.env<-` <- function(fun, value) {
    .setenv(fun, value)
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

