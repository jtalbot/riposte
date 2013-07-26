
.Internal <- function(call) {
    call <- .pr_expr(.getenv(NULL), 'call')
    internal <- getRegisteredNamespace('internal') 
    if(.env_exists(internal, strip(call[[1]]))) {
        call[[1]] <- internal[[strip(call[[1]])]]
        promise('call', call, .frame(1L)[[1L]], .getenv(NULL))
        call
    }
    else {
        .stop(.concat(list("there is no .Internal function '", call[[1]], "'")))
    }
}
