
sort <- function(x, decreasing) {

    if(length(x) > 1L) {
        x[.Riposte('order', FALSE, .isTRUE(decreasing), list(x))]
    }
    else {
        x
    }
}


