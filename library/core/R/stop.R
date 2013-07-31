
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

        .cat(message, '\n') 
        
        n <- 1L
        repeat {
            call <- .frame(n)[[2L]]
            if(!is.null(call))
                .cat(n, ': ', .deparse.call(call),'\n')

            if(is.null(.frame(n)[[6L]]))
                break
            if(n > 10L)
                break

            n <- n+1L
        }

        .stop()
    }

    stop <<- function(include.call, message) {
        # R seems to ignore include.call, what's up?
        call <- .frame(2L)[[2L]]
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

