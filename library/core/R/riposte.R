
repl <- function() {
    promise('print', quote(print(.Last.value)), globalenv(), .getenv(NULL))
    print
   
    if(     (identical(options('warn')[[1]], 0L) || 
            is.null(options('warn')[[1]])) &&
            length(last.warning) > 0L) {
        if(!is.null(baseenv())) {
            env <- baseenv()
            env[['last.warning']] <- last.warning
            last.warning <<- NULL
        }
        promise('warnings', quote(print(warnings())), globalenv(), .getenv(NULL))
        warnings
    }
    NULL
}

# This is hidden by print in the base package.
# Include here so we can have minimal printing
# functionality with only the core package.
print <- function(x, ...) {
    .cat(.format(x),'\n')
    .invisible(x)
}

warnings <- function() {
    w <- last.warning
    last.warning <<- NULL
    w
}

trace.config <- function(trace=0) .External('trace.config', trace)

library <- function(., parent.env) {
    e <- .env_new(parent.env)
    setRegisteredNamespace(.pconcat('',.), e)

    if(identical(.,'base')) {
        e[['.BaseNamespaceEnv']] <- e
    }

    .External('library', e, .)
        
    if(identical(.,'base')) {
        e[['.Options']] <- options()
    }

    # export names and place in search path
    names <- .env_names(e)
    exported <- names[grepl('^[^\\.]', names, FALSE, '', FALSE, FALSE, TRUE, FALSE)]
    .attach(.export(.pconcat('package:',.), e, exported))

    e
}

`::` <- function(a, b) {
    getRegisteredNamespace(strip(.pr_expr(.getenv(NULL), 'a')))[[strip(.pr_expr(.getenv(NULL), 'b'))]]
}

cummean <- function(x) UseGroupMethod('cummean', 'Math', x)
cummean.default <- function(x)
    .Scan.double('mean', x)

hypot <- function(x, y) .ArithBinary2('hypot_map', x, y)

time <- function(x) {
    start <- proc.time()
    x
    proc.time()-start
}

source <- function(file, eval.env) {
    p <- .External('source', file)
    promise('q', p, eval.env, .getenv(NULL))
    q
}

# this is called when interpreter bytecodes fail
# should propagate error nicely
`__stop__` <- function(message, call = NULL) {
    e <- list(message = message, call = call)
    attr(e, 'class') <- c('error', 'condition')
    #.signalCondition(e, message, call)
    .dfltStop(message, call)
}

