
args <- function(name) {
    if(is.character(name)) {
        name <- get(name, .frame(1L)[[1L]], 'closure', TRUE)
    }

    do.call('function', c(name[['formals']], NULL), .getenv(NULL))
}
