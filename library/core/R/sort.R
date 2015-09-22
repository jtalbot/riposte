
sort <- function(x, decreasing) {

    if(length(x) > 1L) {
        x[.Riposte('order', FALSE, decreasing===TRUE, list(x))]
    }
    else {
        x
    }
}

qsort <- function(x, index.return) {
    if(length(x) > 1L) {
        x[.Riposte('order', FALSE, FALSE, list(x))]
    }
    else {
        x
    }
}

psort <- function(x, partial) {
    # TODO: implement partial sorting
    if(length(x) > 1L) {
        x[.Riposte('order', FALSE, FALSE, list(x))]
    }
    else {
        x
    }
}

