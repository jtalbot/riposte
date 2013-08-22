
# Implements .Internal(paste)
paste <- function(items, sep, collapse) 
{
    r <- vector('character',0)
    if(length(items) > 0) {
        sep <- as.character.default(sep)
        r <- as.character.default(items[[1]])
        for(i in 1L+seq_len(length(items)-1L)) {
            r <- .pconcat(r, sep)
            r <- .pconcat(ifelse(length(r) == 0L && length(items[[i]]) > 0L,
                    '', r), 
                ifelse(length(r) > 0L && length(items[[i]])==0L, 
                    '', as.character.default(items[[i]])))
        }
    }
    if(!is.null(collapse)) {
        if(length(collapse) != 1L)
            stop("invalid 'collapse' argument")
        collapse <- as.character.default(collapse)
        r <- .concat(.pconcat(r, c(rep_len(collapse, pmax(0L,length(r)-1L)),'')))
    }
    r
}

# Implements .Internal(paste0)
paste0 <- function(items, collapse)
{
    paste(items, "", collapse)
}
