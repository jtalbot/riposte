
charToRaw <- function(x) {
    if(length(x) != 1L)
        .stop('argument should be a character vector of length 1')
    .External('charToRaw', as.character(x))
}

rawToChar <- function(x, multiple) {
    if(multiple)
        .External('rawToCharacters', as.raw(x))
    else
        .External('rawToChar', as.raw(x))
}

