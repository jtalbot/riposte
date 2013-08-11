
args <- function(name) {
    if(is.character.default(name)) {
        name <- get(name, .frame(1L), 'closure', TRUE)
    }

    do.call('function', c(name[['formals']], NULL), .getenv(NULL))
}
