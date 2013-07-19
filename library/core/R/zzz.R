
getRegisteredNamespace <- NULL
setRegisteredNamespace <- NULL

(function() {
    namespaces <- .env_new(environment(environment(.getenv(NULL))))
    
    namespaces[['empty']] <- environment(environment(.getenv(NULL)))
    namespaces[['core']] <- environment(.getenv(NULL))
    
    getRegisteredNamespace <<- function(name) {
        return( namespaces[[strip(name)]] )
    }

    setRegisteredNamespace <<- function(name, env) {
        namespaces[[strip(name)]] <- env

        if(!.env_exists(env, '.__NAMESPACE__.'))
            env[['.__NAMESPACE__.']] <- NULL
    }
})()

attach(.getenv(NULL))
