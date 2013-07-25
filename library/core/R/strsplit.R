
strsplit <- function(text, split, fixed, perl, useBytes)
{
    splits <- gregexpr(split, text, FALSE, perl, fixed, useBytes)

    # TODO: figure out how to vectorize this
    for(i in seq_len(length(text))) {
        s <- strip(splits[[i]])
        l <- attr(splits[[i]], 'match.length')

        r <- vector('character', 0)
        start <- 1L
        for(j in seq_len(length(s))) {
            r[[length(r)+1L]] <- substr(text[[i]], start, s[[j]]-1L)
            start <- s[[j]] + l[[j]]
        }
        if(start < nchar(text[[i]])) {
            r[[length(r)+1L]] <- substr(text[[i]], start, nchar(text[[i]]))
        }
        splits[[i]] <- r
    }
    
    splits
}

