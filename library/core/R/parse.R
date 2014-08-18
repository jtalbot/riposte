
parse <- function(file, n, text, prompt, srcfile, encoding) {
    if(strip(file) != stdin()) {
        text <- paste(list(readLines(file, -1L, FALSE, FALSE, NULL)), '', '\n')
    }

    if(is.character(srcfile)) {
        # do nothing
    }
    else if(inherits(srcfile, 'srcfile', FALSE)) {
        srcfile <- srcfile$filename
    }
    else {
        srcfile <- ''
    }

    .Riposte('parse', as.character.default(text), as.integer(n), as.character.default(srcfile))
}
