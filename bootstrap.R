
# Bootstraps Riposte off of the R base library *sources*
# (not the compiled versions)
# The base libraries are only avilable under a more restrictive
# license than Riposte, so they are distributed separately.
# You can get them from CRAN.

(function() {
    # Set up the base namespace
    namespace.base <- .env_new(emptyenv())
    internal::registerNamespace('baseenv',namespace.base)
    internal::registerNamespace('base',namespace.base)
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

    namespace.base$.Platform <- list(
        OS.type="unix",
        file.sep="/",
        dynlib.ext=".so",
        GUI="X11",
        endian="little",
        pkgType="mac.binary.mavericks",
        path.sep=":",
        r_arch=""
        )

    # Set up some other variables that R must set somewhere else
    namespace.base[['.Options']] <- internal::options()

    internal::options(prompt="> ")
    internal::options(continue="+ ")
    internal::options(verbose=FALSE)
    internal::options(width=80L)
    internal::options(ts.eps=1e-05)
    internal::options(keep.source=internal::Sys.getenv('R_KEEP_PKG_SOURCE','')=='yes')
    internal::options(keep.source.pkgs=internal::Sys.getenv('R_KEEP_PKG_SOURCE','')=='yes')

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

    # Run the base stuffs
    source('library/base/R/base', namespace.base)
    source('library/base/R/Rprofile', namespace.base)

    # Expose all the base namespace functions in the search path
    names <- internal::ls(namespace.base, TRUE)
    base[names] <- namespace.base[names]
    internal::registerNamespace('baseenv',base)

    # Load the Riposte version of the R API 
    internal::dyn.load('libR.dylib', TRUE, TRUE, '')
    #internal::dyn.load('libRblas.dylib', TRUE, TRUE, '')

    invisible("Loaded base library")
})()

invisible(.First.sys())

