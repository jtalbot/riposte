
sprintf <- function(...) {
    args <- list(...)

    args[[1]] <- as.character(args[[1]])
    specs <- .Riposte('printf_parse', args[[1]][[1]])

    if(length(specs)+1L != length(args)) {
        print("sprintf called with the wrong number of arguments")
        print(args[[1]])
        print(length(args))
    }

    a <- list(args[[1]], list(specs))

    for(i in seq_along(args)) {
        if(i > 1L) {
            if(!is.na(specs[[i-1L]])) {
                func <- .pconcat('as.', specs[[i-1L]]) 
                call <- as.call(list(func, args[[i]]))
                promise('p', call, .frame(2L), .getenv(NULL))
                a[[i+1L]] <- p
            }
        }
    }

    .Map('sprintf_map', a, 'character')[[1]]
}

printf_parse <- function(format) {
    .Riposte('printf_parse', as.character(format))
}
