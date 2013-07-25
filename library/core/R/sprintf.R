
sprintf <- function(...) {
    .Map('sprintf_map', list(...), 'character')[[1]]
}

