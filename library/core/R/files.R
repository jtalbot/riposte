
file.exists <- function(files) {
    .Map('fileexists_map', list(as.character.default(files)), 'logical')[[1]]
}

dir.exists <- function(dirs) {
    .Map('direxists_map', list(as.character.default(dirs)), 'logical')[[1]]
}

