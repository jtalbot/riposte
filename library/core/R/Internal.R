
.Internal <- function(call) {
    call <- .pr_expr(.getenv(NULL), 'call')
    internal <- getRegisteredNamespace('internal') 
    if(.env_has(internal, strip(call[[1]]))) {
        call[[1]] <- internal[[strip(call[[1]])]]
        promise('call', call, .frame(1L), .getenv(NULL))
        call
    }
    else {
        .stop(.concat(list("there is no .Internal function '", call[[1]], "'")), .frame(1L)[['.__call__.']])
    }
}

