
basename <- function(path) {
    .Map('basename_map', list(as.character(path)), 'character')[[1L]]
}

dirname <- function(path) {
    .Map('dirname_map', list(as.character(path)), 'character')[[1L]]
}

