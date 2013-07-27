
pmatch <- function(x, table, nomatch, duplicates.ok)
{
    # TODO: clean this up. Can it share code with match.call?

    nomatch <- as.integer(nomatch)

    if(duplicates.ok) {
        r <- match(x, table, 0L, '')
        for(i in seq_len(length(x))) {
            m <- which(substr(table,1L,nchar(x[[i]])) == x[[i]])
            if(length(m) > 0L)
                r[[i]] <- m[[1L]]
        }
        r[i==0L] <- nomatch
    }
    else {
        r <- match(x, table, 0L, '')
        complete <- .match(seq_len(length(table)), x) > 0L

        for(i in seq_len(length(x))) {
            if(r[[i]] == 0L && x[[i]] != '') {
                m <- (substr(table,1L,nchar(x[[i]])) == x[[i]])
                m <- m & !complete

                if(sum(m) == 1L)
                    r[[i]] <- which(m)[[1L]]
                else
                    r[[i]] <- nomatch
            }
        }
    }
    r
}

