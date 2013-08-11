
# Bootstraps Riposte off of the R base library *sources*
# (not the compiled versions)
# The base libraries are only avilable under a more restrictive
# license than Riposte, so they are distributed separately.
# You can get them from CRAN.

(function() {
    library('base', internal::getRegisteredNamespace('core'))
    core::source('library/profile/Rprofile.unix', baseenv())
    core::source('library/profile/Common.R', baseenv())

    # set some things that I don't yet know where R sets them
    options(verbose=FALSE)
    options(width=80L)
    options(ts.eps=1e-05)

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
