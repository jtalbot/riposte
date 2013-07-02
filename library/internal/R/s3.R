
inherits <- function(x, what, which) {
    if(which)
        match(what, class(x), 0, NULL)
    else
        any(match(what, class(x), 0, NULL) > 0L)
}
