
sprintf <- function(...) {
    args <- list(...)

    args[[1]] <- as.character(args[[1]])
    specs <- .External('printf_parse', args[[1]][[1]])

    if(length(specs)+1L != length(args)) {
        print("sprintf called with the wrong number of arguments")
        print(args[[1]])
        print(length(args))
    }

    for(i in seq_along(args)) {
        if(i > 1L) {
            if(!is.na(specs[[i-1L]])) {
                func <- .pconcat('as.', specs[[i-1L]]) 
                call <- as.call(list(func, args[[i]]))
                promise('p', call, .frame(2L), .getenv(NULL))
                args[[i]] <- p
            }
        }
    }
    .Map('sprintf_map', args, 'character')[[1]]
}

printf_parse <- function(format) {
    .External('printf_parse', as.character(format))
}
