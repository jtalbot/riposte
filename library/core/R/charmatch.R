
charmatch <- function(x, table, nomatch)
{
    nomatch <- as.integer(nomatch)
    r <- vector('integer', 0)
    for(i in seq_len(length(x))) {
        m <- which(table == x[[i]])
        if(length(m) == 1L)
            r[[i]] <- m[[1L]]
        else if(length(m) == 0L) { 
            m <- which(substr(table,1L,nchar(x[[i]])) == x[[i]])
            if(length(m) == 1L)
                r[[i]] <- m[[1L]]
            else if(length(m) == 0L)
                r[[i]] <- nomatch
            else
                r[[i]] <- 0L
        }
        else {
            r[[i]] <- 0L
        }
    }
    r
}

