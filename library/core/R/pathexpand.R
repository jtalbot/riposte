
path.expand <- function(path) {
    .Map('pathexpand_map', list(path), 'character')[[1]]
}

