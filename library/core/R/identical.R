
identical <- function(x, y, num.eq, single.NA, 
    attrib.as.set, ignore.bytecode, ignore.environment) {

    if(.type(x) != .type(y))
        return(FALSE)

    if(is.atomic(x) || is.list(x)) {
        length(x) == length(y) &&
        all(((!is.na(x) & !is.na(y) & (x == y))) |
            (is.na(x) & is.na(y)))
    }
    else if(is.environment(x)) {
        x == y
    }
    else {
        .stop(sprintf("identical NYI for %s", .type(x)))
    }
}

