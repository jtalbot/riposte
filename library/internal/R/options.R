
.Options <- list()

options <- function(...) {
    if(...() == 0L) {
        .Options
    }
    else {
        args <- list(...)
        if(is.null(names(args)) && length(args) == 1 && is.list(args[[1]]))
            args <- args[[1]]

        # split into named and unnamed parameters
        n <- names(args)
        if(is.null(n))
            n <- rep('', length(args))

        nargs <- args[n != '']
        .Options[names(nargs)] <- strip(args)
        
        uargs <- .Options[strip(args)[n == '']]
        
        c(nargs,uargs)
    }
}

