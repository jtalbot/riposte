
last.warning <- NULL

warning <- function(include.call, immediate, no.breaks, msg) {
    
    level <- options('warn')[[1]]
    if(is.null(level))
        level <- 0L

    if(identical(level, 2L))
        stop(include.call, msg)

    if(identical(level, 1L) || immediate) {
        if(.isTRUE(include.call))
            msg <- sprintf("Warning in %s :\n  %s", 
                .format.call(.frame(2L)[['.__call__.']]), msg)
        .cat(msg,'\n')
    }
    else if(identical(level, 0L)) {
        if(.isTRUE(include.call))
            call <- .frame(2L)[['.__call__.']]
        else
            call <- NULL

        end <- length(last.warning)+1L
        last.warning[[end]] <<- call
        names(last.warning)[[end]] <<- msg 
    }

    msg
}

printDeferredWarnings <- function() {
    promise('print', quote(print(warnings())), globalenv(), .getenv(NULL))
    print
    NULL
}

warnings <- NULL
(function() {
    most.recent.warning <- NULL

    warnings <<- function() {
        if(!is.null(last.warning)) {
            most.recent.warning <<- last.warning
            last.warning <<- NULL
        }
        most.recent.warning
    }
})()
