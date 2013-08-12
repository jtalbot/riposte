
# Bootstraps Riposte off of the R base library *sources*
# (not the compiled versions)
# The base libraries are only avilable under a more restrictive
# license than Riposte, so they are distributed separately.
# You can get them from CRAN.

(function() {
    library('base', internal::getRegisteredNamespace('core'))

    # all primitive functions are also available in the base library
    core <- internal::getRegisteredNamespace('core')
    names <- ls(core)
    base <- internal::getRegisteredNamespace('base')
    base[names] <- core[names]

    core::source('library/profile/Rprofile.unix', baseenv())
    core::source('library/profile/Common.R', baseenv())

    # set some things that I don't yet know where R sets them
    options(verbose=FALSE)
    options(width=80L)
    options(ts.eps=1e-05)

    base$.Machine <- list(
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

    load <- function(name) {
        env <- core::library(name, baseenv())
        export <- ls(env)
        export <- export[!grepl('^\\.', export)]
        internal::setRegisteredNamespace(name,
            .make.namespace(name, env, export))
    }

    load('tools')
    load('utils')
    load('stats')
    load('datasets')
    load('methods')
    load('grDevices')
    load('graphics')

    "Loaded base library"
})()
