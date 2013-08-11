
do.call <- function(what, args, envir) {
    if(is.function(what)) {
        # do nothing
    }
    else if(is.character.default(what)) {
        what <- as.name(what)
    }
    # otherwise, assume it's an expression that will evaluate to a function

    do <- as.call(c(what, as.vector(args, 'list')))

    promise('result', do, envir, .getenv(NULL))
    result
}

