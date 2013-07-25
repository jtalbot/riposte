
nchar <- function(x, type = "chars", allowNA = FALSE) {
    .Map.integer('nchar_map', as.character(x))
}

nzchar <- function(x) .Map.logical('nzchar_map', as.character(x))

