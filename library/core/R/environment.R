
environment <- function(fun) {
    if(is.null(fun)) {
        # Note the 2. This is designed to be used from 
        # inside the base library's environment function.
        # If called directly, it will give wrong results.
        # R's .Internal(environment(NULL)) function has
        # the same issue.
        parent.frame(2)
    }
    else
        .getenv(fun)
}

`environment<-` <- function(fun, value) {
    if(!is.environment(value))
        stop("replacement object is not an environment")

    .setenv(fun, value)
}

# is.environment defined in coerce.R

globalenv <- function() {
    # TODO: could replace with looking up in the namespace list...
    .External(globalEnvironment())
}

emptyenv <- function() {
    .External(emptyEnvironment())
}

baseenv <- function() {
    stop("NYI: baseenv needs to look up the base environment in the namespace list")
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

