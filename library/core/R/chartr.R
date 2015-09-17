
chartr <- function(old, new, x) {
    old <- as.character(old)
    new <- as.character(new)
    if(length(old) == 0L)
        .stop("invalid 'old' argument")
    if(length(new) == 0L)
        .stop("invalid 'new' argument")
    old <- old[[1L]]
    new <- new[[1L]]

    if(.nchar(old) != .nchar(new))
        .stop("'old' and 'new' are not the same length")

    .Map('chartr_map',
            list(x,
            .Riposte('chartr_compile',
                old[[1L]],
                new[[1L]])),
            'character')[[1L]]
}

tolower <- function(x) {
    chartr('ABCDEFGHIJKLMNOPQRSTUVWXYZ',
           'abcdefghijklmnopqrstuvwxyz',
            x)
}

toupper <- function(x) {
    chartr('abcdefghijklmnopqrstuvwxyz',
           'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
           x)
}

