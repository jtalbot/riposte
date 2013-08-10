
copyDFattr <- function(xx, x) {
    attributes(x) <- attributes(xx)
}

shortRowNames <- function(x, type) {
    if(type == 0L)
        attr(x, 'row.names')
    else    
        length(attr(x, 'row.names'))
}
