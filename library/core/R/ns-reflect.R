
# in zzz.R for now...

getRegisteredNamespace <- NULL
isRegisteredNamespace <- NULL
registerNamespace <- NULL
getNamespaceRegistry <- NULL

(function() {
    namespaces <- .env_new(.getenv(.getenv(.getenv(NULL))))

    namespaces[['empty']] <- .getenv(.getenv(.getenv(NULL)))

    getRegisteredNamespace <<- function(name) {
        return( namespaces[[strip(name)]] )
    }

    isRegisteredNamespace <<- function(name) {
        .env_has(namespaces, strip(name))
    }

    registerNamespace <<- function(name, env) {
        attr(env,'name') <- .pconcat('namespace:',strip(name))

        namespaces[[strip(name)]] <<- env

        if(!.env_has(env, '.__NAMESPACE__.'))
            env[['.__NAMESPACE__.']] <- NULL

        env
    }

    getNamespaceRegistry <<- function() {
        namespaces
    }
})()

