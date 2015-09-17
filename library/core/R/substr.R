
substr <- function(x, start, stop) {
    text <- as.character.default(x)
   
    start <- pmin(pmax(as.integer(start)-1L, 0L), .nchar(text))
    length <- pmax(as.integer(stop)-start, 0L)
 
    .Map('substr_map', list(text, start, length), 'character')[[1]]
}

`substr<-` <- function(x, start, stop, value) {
    text <- as.character.default(x)
    repl <- as.character.default(value)
 
    start <- pmin(pmax(as.integer(start)-1L, 0L), .nchar(text))
    length <- pmin(pmax(as.integer(stop)-start, 0L), .nchar(repl))
 
    repl <- .Map('substr_map', list(repl, 0L, length), 'character')[[1]]
    .Map('substrassign_map', list(text, start, length, repl), 'character')[[1]]
}

