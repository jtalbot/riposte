
trace.config <- function(trace=0) .External(trace.config(trace))

library <- function(.) {
    e <- .env_new(getRegisteredNamespace('core'))
    setRegisteredNamespace(.pconcat('',.), e)

    # TODO: a better way to handle this case?
    if(identical(.,'base')) {
        e[['.BaseNamespaceEnv']] <- e
    }

    .External(library(e, .))
        
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
    p <- .External(source(file))
    promise('q', p, eval.env, .getenv(NULL))
    q
}
