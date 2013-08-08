
.find <- function(name, env, type)
{
    while(env != emptyenv())
    {
        if(.env_exists(env, name) && (type == "any" || .type(env[[name]]) == type))
            return(as.name(name))
        else {
            env <- .getenv(env)
        }
    }
    NULL
}

.class <- function(x) {
    r <- attr(x, 'class')
    if(is.null(r)) {
        r <- switch(.type(x),
            integer     = .characters('integer', 'numeric'),
            double      = .characters('double', 'numeric'),
            closure     = .characters('function'),
            typeof(x)
        )
        if(length(attr(x,'dim'))==2L) {
            r <- .c('matrix', r)
        }
        else if(length(attr(x,'dim'))>0L) {
            r <- .c('array', r)
        }
    }
    r
}

.resolve.generic.call <- function(generic, generic.name, classlist, call, callenv, defenv, match.first, match.default) {
    targets <- classlist
    if(match.default)
        targets[[length(targets)+1L]] <- 'default'

    searchlist <- .pconcat(.pconcat(generic, '.'), targets)

    for(i in seq_len(length(searchlist))) {
        if(i == 1 && !match.first)
            next

        n <- searchlist[[i]]
        fn <- .find(n, callenv, "closure")

        if(!is.null(fn)) {
            ncall <- attr(call, 'names')
            call <- strip(call) 
            
            call[[1L]] <- fn

            if(i <= length(classlist))
                newclasslist <- classlist[i:length(classlist)]
            else
                newclasslist <- NULL
            if(i != 1L)
                attr(newclasslist, 'previous') <- classlist
            
            call[[length(call)+1L]] <-
                list(`.Generic`=generic.name,
                     `.Method`=n,
                     `.Class`=newclasslist,
                     `.GenericCallEnv`=callenv,
                     `.GenericDefEnv`=defenv,
                     `.Group`=generic)
           
            if(is.null(ncall)) ncall <- rep.int('', length(call))
            ncall[[length(call)]] <- '__extraArgs__'
            attr(call, 'names') <- ncall
            attr(call, 'class') <- 'call'
            return(call)
        }
    }
    return(NULL)
}

UseMethod <- function(generic, object)
{
    if(missing(object)) {
        formals <- attr(.frame(1L)[[3L]][['formals']], 'names')
        if(length(formals) == 0)
            object <- NULL
        else {
            f <- formals[[1L]]
            if(f == '...')
                f <- '..1'
            mcall <- call('missing', as.name(f))
            promise('miss', mcall, .frame(1L)[[1L]], .getenv(NULL))
            if(miss)
                object <- NULL
            else
                promise('object', as.name(f), .frame(1L)[[1L]], .getenv(NULL))
        }
    }

    call <- .resolve.generic.call(
        generic,
        generic, 
        .class(object), 
        .frame(1L)[[2L]], 
        .frame(2L)[[1L]], 
        .getenv(.frame(1L)[[3L]]),
        TRUE,
        TRUE
        )

    if(!is.null(call)) {
        promise('p', call, .frame(2L)[[1L]], .getenv(NULL))
        return(p)
    }
    else {
        .stop(sprintf("no applicable method for '%s' applied to an object of class \"%s\"", generic, .class(object)[[1]]))
    }
}

NextMethod <- function(generic, object, ...) {
    if(missing(generic))
        generic <- parent.frame(1L)[['.Generic']]

    formals <- names(sys.function(sys.parent(1))[['formals']])

    call <- list(as.name(generic))
    names <- ''
    for(i in seq_len(length(formals))) {
        call[[length(call)+1L]] <- as.name(formals[[i]])
        names[[length(names)+1L]] <- formals[[i]]
    }
    names(call) <- names
    class(call) <- 'call'

    class <- .frame(2L)[[1L]][['.Class']]
    callenv <- .frame(2L)[[1L]][['.GenericCallEnv']]
    defenv <- .frame(2L)[[1L]][['.GenericDefEnv']]

    call <- .resolve.generic.call(
        generic,
        generic,
        class, 
        call,
        callenv,
        defenv,
        FALSE,
        TRUE)
    
    if(!is.null(call)) {
        promise('p', call, .frame(2L)[[1L]], .getenv(NULL))
        return(p)
    }
    else {
        .stop("no method to invoke")
    }
}
