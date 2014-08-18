
sort <- function(x, decreasing) {

    if(length(x) > 1L) {
        x[.External('order', FALSE, .isTRUE(decreasing), list(x))]
    }
    else {
        x
    }
}


