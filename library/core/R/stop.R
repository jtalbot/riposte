
stop <- NULL
geterrmessage <- NULL
seterrmessage <- NULL

(function() {

    errmessage <- ""

    stop <<- function(include.call, message) {
        errmessage <<- .pconcat(message, '\n')

        if(include.call)
            message <- sprintf("Error in %s :\n  %s", 
                format.call(.frame(2L)[[2L]]), message)
           
        n <- 1L
        repeat {
            call <- .frame(n)[[2L]]
            if(!is.null(call))
                .cat(n, ': ', format.call(call),'\n')

            if(is.null(.frame(n)[[6L]]))
                break
            if(n > 10L)
                break

            n <- n+1L
        }

        .cat(message, '\n') 

        .External(stop(message))
    }

    geterrmessage <<- function() {
        errmessage
    }

    seterrmessage <<- function(msg) {
        errmessage <<- msg
    }
})()

