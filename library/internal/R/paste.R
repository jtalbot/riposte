
# Implements .Internal(paste)
paste <- function(items, sep, collapse) 
{
    r <- vector('character',0)
    if(length(items) > 0) {
        sep <- as.character(sep)
        r <- items[[1]]
        for(i in seq(2L,1L,length(items)-1L)) {
            r <- .pconcat(r, sep)
            r <- .pconcat(r, items[[i]])
        }
    }
    if(!is.null(collapse) && length(r) > 1L) {
        if(length(collapse) != 1)
            stop("invalid 'collapse' argument")
        collapse <- as.character(collapse)
        r <- .concat(.pconcat(r, c(rep(collapse, length(r)-1L),'')))
    }
    r
}

# Implements .Internal(paste0)
paste0 <- function(items, collapse)
{
    paste(items, "", collapse)
}
