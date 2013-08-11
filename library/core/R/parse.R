
parse <- function(file, n, text, prompt, srcfile, encoding) {
    if(strip(file) != stdin()) {
        text <- paste(readLines(file, warn=FALSE), '', '\n')
    }
    .External('parse', as.character.default(text), as.integer(n), as.character.default(srcfile))
}
