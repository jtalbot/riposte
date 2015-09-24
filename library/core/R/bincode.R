
# TODO: this could be made O(log(n)) in the length of breaks
# with a binary search.

bincode <- function(x, breaks, right, include.lowest)
{
    if(right)
    {
        if(include.lowest)
            idx <- ifelse(x >= breaks[[1L]], 1L, NA)
        else
            idx <- ifelse(x > breaks[[1L]], 1L, NA)

        for(i in seq_len(length(breaks)-2L)+1L)
        {
            idx[x > breaks[[i]] ] <- i
        }

        idx[x > breaks[[length(breaks)]] ] <- NA_integer_

        idx
    }
    else
    {
        idx <- ifelse(x >= breaks[[1L]], 1L, NA)

        for(i in seq_len(length(breaks)-2L)+1L)
        {
            idx[x >= breaks[[i]] ] <- i
        }

        if(include.lowest)
            idx[x > breaks[[length(breaks)]] ] <- NA_integer_
        else
            idx[x >= breaks[[length(breaks)]] ] <- NA_integer_

        idx
    }
}
