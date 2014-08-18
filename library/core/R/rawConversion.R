
charToRaw <- function(x) {
    if(length(x) != 1L)
        .stop('argument should be a character vector of length 1')
    .Riposte('charToRaw', as.character(x))
}

rawToChar <- function(x, multiple) {
    if(multiple)
        .Riposte('rawToCharacters', as.raw(x))
    else
        .Riposte('rawToChar', as.raw(x))
}

