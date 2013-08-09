
isNamespaceEnv <- function(ns) {
    !is.nil(.get(ns, '.__NAMESPACE__.'))
}

