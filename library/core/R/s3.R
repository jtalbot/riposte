
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
    r
}

.resolve.generic.call <- function(generic, classlist, call, callenv, defenv, match.first) {
    targets <- classlist
    targets[[length(targets)+1L]] <- 'default'

    searchlist <- .pconcat(.pconcat(generic, '.'), targets)

    for(i in seq_len(length(searchlist))) {
        if(i == 1 && !match.first)
            next

        n <- searchlist[[i]]
        fn <- .find(n, callenv, "closure")

        if(!is.null(fn)) {
            # TODO: clean this up...
            ncall <- names(call)
            call <- strip(call) 
            
            call[[1L]] <- fn

            if(i <= length(classlist))
                .Class <- classlist[i:length(classlist)]
            else
                .Class <- NULL
            if(i != 1L)
                attr(.Class, 'previous') <- classlist

            call[[length(call)+1L]] <-
                list(`.Generic`=generic,
                     `.Method`=n,
                     `.Class`=.Class,
                     `.GenericCallEnv`=callenv,
                     `.GenericDefEnv`=defenv)
            
            if(is.null(ncall)) ncall <- rep('', length(call))
            ncall[[length(call)]] <- '__extraArgs__'
            names(call) <- ncall
            
            class(call) <- 'call'
            return(call)
        }
    }
    return(NULL)
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
            mcall <- call('missing', as.name(f))
            promise('miss', mcall, parent.frame(), parent.frame(0))
            if(miss)
                object <- NULL
            else
                promise('object', as.name(f), parent.frame(), parent.frame(0))
        }
    }
    call <- .resolve.generic.call(
        generic, 
        .class(object), 
        sys.call(sys.parent(1)), 
        parent.frame(2), 
        environment(sys.function(sys.parent(1))),
        TRUE
        )

    if(!is.null(call)) {
        promise('p', call, parent.frame(2), parent.frame(0))
        return(p)
    }
    else {
        stop(.concat(list("no applicable method for '", generic,"' applied to and object of class \"", .class(object)[[1]], "\"")))
    }
}

