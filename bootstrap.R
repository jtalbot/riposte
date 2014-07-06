
# Bootstraps Riposte off of the R base library *sources*
# (not the compiled versions)
# The base libraries are only avilable under a more restrictive
# license than Riposte, so they are distributed separately.
# You can get them from CRAN.

(function() {
    # Set up the base namespace
    namespace.base <- .env_new(emptyenv())
    attr(namespace.base,'name') <- 'namespace:base'
    internal::registerNamespace('base',namespace.base)
    internal::registerNamespace('baseenv',namespace.base)
    namespace.base[['.BaseNamespaceEnv']] <- namespace.base

    # Riposte's primitive package holds what are primitive functions in GNU R.
    # All primitive functions are also available in the base library
    primitive <- internal::getRegisteredNamespace('primitive')
    names <- internal::ls(primitive, TRUE)
    namespace.base[names] <- primitive[names]

    # Create the base package to go in the search path and put it there
    # replacing the primitive package which was just there to bootstrap
    base <- .env_new(emptyenv())
    attr(base,'name') <- 'base'
    base[names] <- primitive[names]
    .setenv(base, emptyenv())
    .setenv(.env_global(), base)

    # Pretend like we've loaded a base DLL (one doesn't really exist)
    internal::dyn.load('base', FALSE, FALSE, '')

    # Run the base stuffs
    source('library/base/R/base', namespace.base)
    source('library/base/R/Rprofile', namespace.base)

    # Set up some other variables that R must set somewhere else
    namespace.base[['.Options']] <- internal::options()

    internal::options(verbose=FALSE)
    internal::options(width=80L)
    internal::options(ts.eps=1e-05)

    namespace.base$.Machine <- list(
        double.eps = 2.22044604925031e-16,
        double.neg.eps = 1.11022302462516e-16,
        double.xmin = 2.2250738585072e-308,
        double.xmax = 1.79769313486232e+308,
        double.base = 2L,
        double.digits = 53L,
        double.rounding = 5L,
        double.guard = 0L,
        double.ulp.digits = -52L,
        double.neg.ulp.digits = -53L,
        double.exponent = 11L,
        double.min.exp = -1022L,
        double.max.exp = 1024L,
        integer.max = 9223372036854775807L,
        sizeof.long = 8L,
        sizeof.longlong = 8L,
        sizeof.longdouble = 16L,
        sizeof.pointer = 8L
        )

    # Expose all the base namespace functions in the search path
    names <- internal::ls(namespace.base, TRUE)
    base[names] <- namespace.base[names]
    internal::registerNamespace('baseenv',base)

    #namespace <- function(name, env) {
    
    #import <- new.env(env)
    #attr(import, 'name') <- paste('imports:',name,sep="")

    #exports <- new.env()
    #exportpatterns <- character(0)
    #S3methods <- matrix(character(0), nrow=0, ncol=3)

    #ns <- new.env(baseenv())
    #ns$useDynLib <- function(library, ..., .registration=FALSE, .fixes="") NULL
    #ns$import <- function(library) {
    #    env <- getRegisteredNamespace(strip(.pr_expr(.getenv(NULL), 'library')))
    #    names <- ls(env)
    #    import[names] <- env[names]
    #}
    #ns$importFrom <- function(library, ...) {
    #    env <- getRegisteredNamespace(strip(.pr_expr(.getenv(NULL), 'library')))
    #    for(i in seq_len(length(`__dots__`))) {
    #        name <- strip(.pr_expr(`__dots__`, i))
    #        import[[name]] <- env[[name]]
    #    }
    #}
    #ns$export <- function(...) {
    #    for(i in seq_len(length(`__dots__`))) {
    #        name <- strip(.pr_expr(`__dots__`, i))
    #        exports[[name]] <- name
    #    }
    #}
    #ns$exportPattern <- function(pattern)
    #    exportpatterns[length(exportpatterns)+1] <- pattern
    #
    #ns$S3method <- function(func, type, f) {
    #    if(missing(f))
    #        f <- paste(func,type,sep='.')
    #    else
    #        f <- strip(.pr_expr(.getenv(NULL), 'f'))
    #    S3methods <- rbind(S3methods, c(func,type,f))
    #}

    #list( ns,
    #        function() import,
    #        function() exports,
    #        function() exportpatterns,
    #        function() S3methods )
    #}

    #load <- function(name) {
    #    ns <- namespace(name, baseenv())
    #    core::source(paste('library/',name, '/NAMESPACE', sep=""), ns[[1L]])
    #    env <- core::library(name, ns[[2L]]())
    #    exports <- ns[[3L]]()
    #    patterns <- ns[[4L]]()
    #    n <- ls(env, TRUE)
    #    for(p in patterns) {
    #        exported <- grepl(p, n)
    #        exports[exported] <- exported
    #    }
    #    env[['.__NAMESPACE__.']] <- new.env(emptyenv())
    #    env[['.__NAMESPACE__.']][['exports']] <- exports
    #    env[['.__NAMESPACE__.']][['S3methods']] <- ns[[5L]]()
    #    attr(env, 'name') <- paste('namespace:', name, sep="")
    #    internal::setRegisteredNamespace(name, env)
    #}

    # base packages

    # the first 5 have to be in this order
#    load('tools')
#    load('utils')
#    load('stats')
#    load('datasets')
#    load('methods')
#    # then, in any order
#    load('grDevices')
#    load('graphics')
#    load('grid')
##    load('parallel')
#    load('splines')
#    load('stats4')

    # recommended
#    load('KernSmooth')
#    load('MASS')
#    load('Matrix')
#    load('boot')
#    load('class')
#    load('cluster')
#    load('codetools')
#    load('foreign')
#    load('lattice')
#    load('mgcv')
#    load('nlme')
#    load('nnet')
#    load('rpart')
#    load('spatial')
#    load('survival')

    "Loaded base library"
})()

.First.sys()

