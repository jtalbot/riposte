
.dfltStop <- NULL
stop <- NULL
geterrmessage <- NULL
seterrmessage <- NULL

(function() {

    errmessage <- ""

    .dfltStop <<- function(message, call) {
        if(!is.null(call))
            message <- sprintf("Error in %s : %s", 
                .deparse.call(call), message)
        else
            message <- sprintf("Error: %s", message)

        .connection.cat.terminal(2L, .pconcat(message, '\n')) 
        
        n <- 1L
        repeat {
            call <- .frame(n)[['__call__']]
            if(!is.null(call))
                .cat(n, ': ', .deparse(call),'\n')

            if(is.null(.frame(n)[['__parent__']]))
                break
            if(n > 100L)
                break

            n <- n+1L
        }

        .stop()
    }

    stop <<- function(include.call, message) {
        # R seems to ignore include.call, what's up?
        call <- .frame(2L)[['__call__']]
        e <- list(message=message, call=call)
        attr(e, 'class') <- c('error', 'condition')

        .signalCondition(e, message, call)
        .dfltStop(message, call)
    }
           
    geterrmessage <<- function() {
        errmessage
    }

    seterrmessage <<- function(msg) {
        errmessage <<- msg
    }
})()

