
missing <- function(...) {
    if(length(`.__call__.`) != 2)
        .stop("missing called with the wrong number of arguments")

    if(`.__call__.`[[2L]] == quote(...))
        is.null(.frame(1L)[['.__names__.']])
    else
        .env_missing(.frame(1L), strip(`.__call__.`[[2L]]))
}

