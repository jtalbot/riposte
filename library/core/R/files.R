
file.exists <- function(files) {
    .Map('fileexists_map', list(as.character(files)), 'logical')[[1]]
}
