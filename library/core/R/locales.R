
Sys.getlocale <- function(category) {
    .Map('getlocale_map', list(as.integer(category)), 'character')[[1]]
}

Sys.setlocale <- function(category, locale) {
    .Map('getlocale_map',
        list(as.character(category), as.integer(locale)), 'character')[[1]]
}
