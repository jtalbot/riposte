
.find.group.method <- function(generic, group, object) {
    call <- .resolve.generic.call(
        generic, 
        generic,
        .class(object), 
        .frame(2L)[['__call__']], 
        .frame(3L), 
        .getenv(.frame(2L)[['__function__']]),
        TRUE,
        FALSE
        )

    if(is.null(call)) {
        # attempt to find matching group
        call <- .resolve.generic.call(
            group,
            generic, 
            .class(object), 
            .frame(2L)[['__call__']], 
            .frame(3L), 
            .getenv(.frame(2L)[['__function__']]),
            TRUE,
            FALSE
            )
    }

    call
}

UseGroupMethod <- function(generic, group, object)
{
    call <- .find.group.method(generic, group, object)

    if(is.null(call)) {
        # finally, match the default
        call <- .resolve.generic.call(
            generic,
            generic, 
            'default', 
            .frame(1L)[['__call__']], 
            .frame(2L), 
            .getenv(.frame(1L)[['__function__']]),
            TRUE,
            FALSE
            )
    }

    if(!is.null(call)) {
        promise('p', call, .frame(2L), .getenv(NULL))
        return(p)
    }
    else {
        .stop(sprintf("no applicable method for '%s' applied to an object of class \"%s\"", generic, .class(object)[[1]]))
    }
}

UseMultiMethod <- function(generic, group, ...)
{
    if(...() == 0L)
        return(NULL)

    args <- list(...)
    
    calls <- list()
    for(i in seq_len(length(args)))
        calls[[i]] <- .find.group.method(generic, group, args[[i]])

    funcs <- as.character.default(
        lapply(calls, function(x) 
            if(is.null(x)) 'NULL' else strip(x[[1]])))

    if(!any(funcs=='NULL') && all(funcs[[1]] == funcs))
        call <- calls[[1]]
    else
        call <- NULL

    if(is.null(call)) {
        #.warning(sprintf("Incompatible methods (%s) for \"%s\"", 
        #    paste(funcs, ', ', ''), generic))

        # finally, match the default
        call <- .resolve.generic.call(
            generic,
            generic, 
            'default', 
            .frame(1L)[['__call__']], 
            .frame(2L), 
            .getenv(.frame(1L)[['__function__']]),
            TRUE,
            FALSE
            )
    }

    if(!is.null(call)) {
        promise('p', call, .frame(2L), .getenv(NULL))
        return(p)
    }
    else {
        .stop(sprintf("no applicable method for '%s' applied to an object of class \"%s\"", generic, .class(object)[[1]]))
    }
}

