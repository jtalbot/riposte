
basename <- function(path) {
    .Map('basename_map', list(as.character.default(path)), 'character')[[1L]]
}

dirname <- function(path) {
    .Map('dirname_map', list(as.character.default(path)), 'character')[[1L]]
}

