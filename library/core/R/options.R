
options <- (function() {

    Options <- .env_new(.getenv(.getenv(NULL)))

    function(...) {
        if(...() == 0L) {
            env2list(Options, TRUE)
        }
        else {
            args <- list(...)
            if(is.null(names(args)) && length(args) == 1 && is.list(args[[1]]))
                args <- args[[1]]

            # split into named and unnamed parameters
            n <- names(args)
            if(!is.null(n)) {
                nargs <- args[n != '']
                Options[names(nargs)] <<- strip(nargs)
            }
            else {
                nargs <- list()
            }

            if(!is.null(n))
                nuargs <- as.character(strip(args)[n == ''])
            else
                nuargs <- as.character(strip(args))
            uargs <- Options[nuargs]
            names(uargs) <- nuargs
 
            c(nargs,uargs)
        }
    }

})()
