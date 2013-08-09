
.list2env <- function(l) {
    n <- names(l)
    if(length(l) > 0 && (!is.character(n) || length(n) != length(l)))
        .stop("names(x) must be a character vector of the same length as x")

    e <- .env_new(emptyenv())
    e[n] <- strip(l)
    e
}

.search.path <- function(n, env) {
    if(n > 0L) {
        env <- .env_global()
        while(n > 0) {
            env <- .getenv(env)
            n <- n-1
        }
        env
    }
    else {
        .getenv(env)
    }
}

as.environment <- function(x)
	switch(.type(x),
		environment=x,
		double=,
		integer=.search.path(as.integer(x), .frame(1L)),
        list=.list2env(x),
		.stop("unsupported cast to environment")) 

