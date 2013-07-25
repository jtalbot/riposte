
.pmatch <- function(x, table, nomatch, ambiguous, duplicates.ok) {
    
    # find complete matches
    r <- match(x, table, NA_integer_, '')
    complete <- rep(FALSE, length(table)) 
    complete[r] <- TRUE

    # find partial matches 
    partial <- rep(FALSE, length(table))
     
    for(i in seq_len(length(x))) {
        if(is.na(r[[i]]) && x[[i]] != '') {
            m <- (substr(table,1L,nchar(x[[i]])) == x[[i]])
            if(!duplicates.ok)
                m <- m & !complete

            if(sum(m) == 0L)
                r[[i]] <- as.integer(nomatch)
            else if(sum(m) > 1L)
                r[[i]] <- as.integer(ambiguous)
            else {
                j <- which(m)
                if(duplicates.ok || partial[[j]] == FALSE) {
                    partial[[j]] <- TRUE
                    r[[i]] <- j
                }
                else {
                    r[r==j] <- as.integer(ambiguous)
                }
            }
        }
    }
}

pmatch <- function(x, table, nomatch, duplicates.ok) {
    .pmatch(x, table, nomatch, nomatch, duplicates.ok)
}

