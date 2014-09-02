
identical <- function(x, y, num.eq, single.NA, 
    attrib.as.set, ignore.bytecode, ignore.environment) {

    if(.type(x) != .type(y))
        return(FALSE)

    if(is.atomic(x)) {
        length(x) == length(y) &&
        all(((!is.na(x) & !is.na(y) & (x == y))) |
            (is.na(x) & is.na(y)))
    }
    else if(is.list(x)) {
        if(length(x) != length(y))
            return(FALSE)
        for(i in seq_along(x)) {
            if(!identical(x[[i]], y[[i]], num.eq, single.NA, attrib.as.set, ignore.bytecode, ignore.environment))
                return(FALSE)
        }
        return(TRUE)
    }
    else {
        x==y
    }
}

