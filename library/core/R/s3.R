
characters <- function(...) as.character(list(...))

.find <- function(name, env, type)
{
    while(env != emptyenv())
    {
        if(.env_exists(env, name) && (type == "any" || .type(env[[name]]) == type))
            return(as.name(name))
        else
            env <- .getenv(env)
    }
    NULL
}

.class <- function(x) {
    r <- attr(x, 'class')
    if(is.null(r)) {
        r <- switch(.type(x),
            integer     = characters('integer', 'numeric'),
            double      = characters('double', 'numeric'),
            closure     = characters('function'),
            typeof(x)
        )
    }
    else {
        
    }
    r[[length(r)+1L]] <- 'default'
    r
}

UseMethod <- function(generic, object, ...)
{
    if(nargs() == 1) {
        formals <- names(sys.function(sys.parent(1))[['formals']])
        if(length(formals) == 0)
            object <- NULL
        else {
            f <- formals[[1L]]
            if(f == '...')
                f <- '..1'
            call <- list(as.name('missing'), as.name(f))
            class(call) <- 'call'
            promise('miss', call, parent.frame(), parent.frame(0))
            if(miss)
                object <- NULL
            else
                promise('object', as.name(f), parent.frame(), parent.frame(0))
        }
    }
    names <- .pconcat(.pconcat(generic, '.'), .class(object))
    for(n in names) {
        fn <- .find(n, parent.frame(2), "closure")
        if(!is.null(fn)) {
            # TODO: clean this up...
            call <- sys.call(sys.parent(1))
            n <- names(call)
            call <- strip(call) 
            call[[1L]] <- fn
            class(call) <- 'call'
            names(call) <- n
            promise('p', call, parent.frame(2), parent.frame(0))
            return(p)
        }
    }
    stop("UseMethod could not find matching generic")
}

