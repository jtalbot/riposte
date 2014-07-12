
isNamespaceEnv <- function(ns) {
    !is.nil(.get(ns, '.__NAMESPACE__.'))
}

importIntoEnv <- function(impenv, impnames, expenv, expnames) {
    impnames <- as.character(impnames)
    expnames <- as.character(expnames)

    if(length(impnames) != length(expnames))
        .stop('length of import and export names must match')

    .External('importIntoEnv', impenv, impnames, expenv, expnames)

    NULL
}

