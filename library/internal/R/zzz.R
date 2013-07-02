
# TODO: actually populate this correctly
.Platform <- list(
    OS.type="unix", 
    file.sep="/", 
    dynlib.ext=".so", 
    GUI="X11", 
    endian="little", 
    pkgType="mac.binary.leopard", 
    path.sep=":", 
    r_arch="x86_64"
    )

# set as a promise so that it doesn't evaluate until the base
# environment is actually loaded
promise('.BaseNamespaceEnv', quote(baseenv()), core::parent.frame(0), core::parent.frame(0))

.GlobalEnv <- globalenv()
