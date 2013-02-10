
# Implements .Internal(paste)
paste <- function(items, sep, collapse) 
{
    r <- .External(
            mapply (items,
                function(...) 
                    .External(paste(list(...), sep))))
	
    if(!is.null(collapse)) 
        .External(paste(r, collapse))
    else 
        .External(unlist(r, TRUE, TRUE))
}

# Implements .Internal(paste0)
paste0 <- function(items, collapse)
{
    paste(items, "", collapse)
}
