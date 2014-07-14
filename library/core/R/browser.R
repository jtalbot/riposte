
browser <- NULL
browserText <- NULL
browserCondition <- NULL
browserSetDebug <- NULL

(function() {
    browserContexts <- vector('list', 0) 

    browser <<- function(text="", condition=NULL, expr=TRUE, skipCalls=0L)
    {
        if(!.isTRUE(expr))
            return

        browserContexts[[length(browserContexts)+1L]] <<-
            list(text, condition)

        on.exit({
            length(browserContexts) <<- length(browserContexts)-1L
            }, FALSE)

        skipCalls <- as.integer(skipCalls)+1L

        repeat {
            .cat(sprintf('Browse[%d]> ',length(browserContexts)))
            cmd <- readLines(stdin(), n=1L)
        
            if(cmd == '') {
                if(.isTRUE(options('browserNLdisabled')))
                    next
                else
                    break
            }

            switch(cmd,
                'c'=,
               'cont'=,
                'n'=break,
                'Q'=.stop(),
                'where'= {
                    n <- skipCalls 
                    repeat {
                        call <- .frame(n)[['.__call__.']]
                        if(!is.null(call))
                            .cat(n, ': ', .format.call(call),'\n')

                        if(is.null(.frame(n)[[6L]]))
                            break

                        n <- n+1L
                    }
                },
                {
                    promise('expr', .External('parse', cmd, -1L, '<stdin>'), 
                        .frame(skipCalls), .getenv(NULL))
                    .cat(.format(expr),'\n')
                })
        }
        NULL
    }

    browserText <<- function(n) {
        if(n <= 0)
            .stop("number of contexts must be positive")
        n <- length(browserContexts)-n+1L
        if(n <= 0)
            .stop("no browser context to query")
        browserContexts[[n]][[1L]]
    }
    
    browserCondition <<- function(n) {
        if(n <= 0)
            .stop("number of contexts must be positive")
        n <- length(browserContexts)-n+1L
        if(n <= 0)
            .stop("no browser context to query")
        browserContexts[[n]][[2L]]
    }

    browserSetDebug <<- function(n) {
        env <- .frame(n+1L)
        .on.exit(env, quote(browser('', NULL, TRUE, 1L)), TRUE)
        NULL
    }
})()
