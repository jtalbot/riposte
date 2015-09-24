
.list2env <- function(l) {
    n <- names(l)
    if(length(l) > 0 && (!is.character.default(n) || length(n) != length(l)))
        .stop("names(x) must be a character vector of the same length as x")

    e <- .env_new(emptyenv())
    e[n] <- strip(l)
    e
}

.search.path <- function(n, env) {
    env <- .env_global()
    while(n > 0) {
        env <- .getenv(env)
        n <- n-1
    }
    env
}

.search.path.name <- function(n, env) {
    env <- .env_global()

    while(env != emptyenv()) {
    
        if(attr(env, 'name') == n)
            return(env)

        env <- .getenv(env)
    }

    .stop(fprintf('no item called "%s" on the search list', n))
}

as.environment <- function(x)
	switch(.type(x),
		environment=x,
		double=,
		integer=if(x >= 0L) .search.path(as.integer(x), .frame(1L)) else .frame(2L),
        character=.search.path.name(as.character(x), .frame(1L)),
        list=.list2env(x),
		.stop("unsupported cast to environment")) 

