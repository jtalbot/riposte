
nchar <- function(x, type, allowNA, keepNA) {
    .Map('nchar_map', list(as.character.default(x)), 'integer')[[1L]]
}

nzchar <- function(x) {
    .Map('nzchar_map', list(as.character.default(x)), 'logical')[[1L]]
}

