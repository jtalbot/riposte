
file.exists <- function(files) {
    .Map('fileexists_map', list(as.character.default(files)), 'logical')[[1]]
}
