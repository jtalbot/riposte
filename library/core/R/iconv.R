
iconv <- function(x, from, to, sub, mark, toRaw) {
    .Map('iconv_map', list(x, from, to, sub), 'character')
}

