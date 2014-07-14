
isNamespaceEnv <- function(ns) {
    !is.nil(.get(ns, '.__NAMESPACE__.'))
}

importIntoEnv <- function(impenv, impnames, expenv, expnames) {
    # The base importIntoEnv code pulls all names in the environment,
    # including .__x__. ones that we're treating specially. Exclude
    # any name that is not a simple string.

    if(length(impnames) != length(expnames))
        .stop('length of import and export names must match')

    for(i in seq_along(impnames)) {
        if(is.character(impnames[[i]]) && is.character(expnames[[i]]))
            .External('importIntoEnv', impenv, impnames[[i]], expenv, expnames[[i]])
    }

    NULL
}

