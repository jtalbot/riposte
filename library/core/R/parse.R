
parse <- function(file, n, text, prompt, srcfile, encoding) {

    if(strip(file) != stdin()) {
        text <- readLines(file, -1L, FALSE, FALSE, NULL)
    }
    text <- paste(list(as.character(text)), '', '\n')

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
