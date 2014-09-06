
.find <- function(name, env, type)
{
    while(env != emptyenv())
    {
        if(!is.nil(.get(env, name)) && (type == "any" || .type(env[[name]]) == type))
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

    methodsTable <- defenv[['.__S3MethodsTable__.']]

    for(i in seq_len(length(searchlist))) {
        if(i == 1 && !match.first)
            next

        n <- searchlist[[i]]
        fn <- .find(n, callenv, "closure")

        if(is.null(fn) && !is.null(methodsTable))
            fn <- methodsTable[[n]]

        if(!is.null(fn)) {
            ncall <- attr(call, 'names')
            call <- strip(call) 
            
            call[[1L]] <- fn

            if(i <= length(classlist))
                newclasslist <- classlist[i+seq_len(length(classlist)-i+1L)-1L]
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
            ncall[[length(call)]] <- '.__extraArgs__.'
            attr(call, 'names') <- ncall
            attr(call, 'class') <- 'call'
            return(call)
        }
    }
    return(NULL)
}

UseMethod <- function(generic, object)
{
    if(.env_missing(NULL,'object')) {
        formals <- attr(.frame(1L)[['.__function__.']][['formals']], 'names')
        if(length(formals) == 0)
            object <- NULL
        else {
            f <- formals[[1L]]
            if(f == '...') {
                f <- '..1'
                mcall <- call('missing', 1L)
            }
            else {
                mcall <- call('missing', as.name(f))
            }
            promise('miss', mcall, .frame(1L), .getenv(NULL))
            if(miss)
                object <- NULL
            else
                promise('object', as.name(f), .frame(1L), .getenv(NULL))
        }
    }
    object

    call <- .resolve.generic.call(
        generic,
        generic,
        .class(object), 
        .frame(1L)[['.__call__.']], 
        .frame(2L), 
        .getenv(.frame(1L)[['.__function__.']]),
        TRUE,
        TRUE
        )

    names <- attr(call,'names')
    call <- strip(call)
    
    if(!is.null(attr(object,'class'))) {
        qq <- list(as.name('quote'), object)
        attr(qq, 'class') <- 'call'
        call[[2L]] <- qq
    }
    else {
        call[[2L]] <- object
    }
    attr(call, 'names') <- names
    attr(call, 'class') <- 'call'

    if(!is.null(call)) {
        promise('p', call, .frame(2L), .getenv(NULL))
        return(p)
    }
    else {
        .stop(sprintf("no applicable method for '%s' applied to an object of class \"%s\"", generic, .class(object)[[1]]))
    }
}

NextMethod <- function(generic, object, ...) {
    if(.env_missing(NULL,'generic'))
        generic <- parent.frame(1L)[['.Generic']]

    class <- .frame(2L)[['.Class']]
    callenv <- .frame(2L)[['.GenericCallEnv']]
    defenv <- .frame(2L)[['.GenericDefEnv']]

    formals <- attr(.frame(2L)[['.__function__.']][['formals']], 'names')
    call <- list()
    names <- NULL
    call[[1]] <- as.name(generic)
    names[[1]] <- ''
    for(i in seq_len(length(formals))) {
        call[[i+1L]] <- as.name(formals[[i]])
        if(formals[[i]] != '...')
            names[[i+1L]] <- formals[[i]]
        else
            names[[i+1L]] <- ''
    }
    attr(call, 'class') <- 'call'
    attr(call, 'names') <- names
 
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
        promise('p', call, .frame(2L), .getenv(NULL))
        return(p)
    }
    else {
        .stop("no method to invoke")
    }
}
