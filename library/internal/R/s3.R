
inherits <- function(x, what, which) {
    if(which)
        match(what, class(x), 0, NULL)
    else
        any(match(what, class(x), 0, NULL) > 0L)
}

NextMethod <- function(generic, object, ...) {
    if(missing(generic))
        generic <- parent.frame(1L)[['.Generic']]

    # TODO: what does setting object do?
    formals <- names(sys.function(sys.parent(1))[['formals']])
    
    call <- list(as.name(generic))
    names <- ''
    for(i in seq_len(length(formals))) {
        call[[length(call)+1L]] <- as.name(formals[[i]])
        names[[length(names)+1L]] <- formals[[i]]
    }
    names(call) <- names
    class(call) <- 'call'

    class <- parent.frame(1L)[['.Class']]
    callenv <- parent.frame(1L)[['.GenericCallEnv']]
    defenv <- parent.frame(1L)[['.GenericDefEnv']]

    call <- .resolve.generic.call(
        generic,
        class, 
        call,
        callenv,
        defenv,
        FALSE)
    
    if(!is.null(call)) {
        promise('p', call, parent.frame(1L), .getenv(NULL))
        return(p)
    }
    else {
        stop("no method to invoke")
    }
}
