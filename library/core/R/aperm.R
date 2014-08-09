
aperm <- function(a, perm, resize) {
    d <- dim(a)
    if(length(d) != length(perm))
        .stop(sprintf("'perm' is of wrong length %d (!= %d)", length(perm), length(d)))
    
    if(all(perm==c(1L,2L)))
        a
    else if(all(perm==c(2L,1L)) && resize)
        t.default(a)
    else
        .stop('aperm called with unsuppored options')
}

