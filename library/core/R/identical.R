
identical <- function(x, y, num.eq, single.NA, 
    attrib.as.set, ignore.bytecode, ignore.environment) {

    # TODO: support arguments
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
        # TODO: support other types
        .stop(sprintf("identical NYI for %s", .type(x)))
    }

    # TODO: check attributes
}

