
Encoding <- function(x) {
    if(is.character.default(x))
        rep_len('unknown', length(x))
    else
        .stop('a character vector argument expected')
}

setEncoding <- function(x, value) {
    if(is.character.default(x))
        x 
    else
        .stop('a character vector argument expected')
    x
}

enc2native <- function(x) {
    x
}

enc2utf8 <- function(x) {
    x
}
