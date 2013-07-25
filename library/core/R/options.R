
#TODO: put this in an environment instead?

.Options <- list(
    add.smooth = TRUE,
    browserNLdisabled = FALSE,
    check.bounds = FALSE,
    CBoundsCheck = FALSE,
    continue = '+ ',
    defaultPackages = .characters('datasets', 'utils', 'grDevices', 'graphics', 'stats', 'methods'),
    deparse.cutoff = 60,
    digits = 7,
    digits.sec = 0,
    echo = TRUE,
    encoding = 'native.enc',
    error = NULL,
    expressions = 5000,
    keep.source = TRUE,
    keep.source.pkgs = FALSE,
    max.print = 99999,
    OutDec = '.',
    pager = function(files, header, title, delete.file) .stop("NYI: options pager"),
    papersize = 'a4',
    pdfviewer = '/usr/bin/open',
    printcmd = 'lpr',
    prompt = '> ',
    rl_word_breaks = " \t\n\"\\'`><=%;,|&{()}",
    scipen = 0,
    show.error.messages = TRUE,
    stringsAsFactors = TRUE,
    texi2dvi = '/usr/bin/texi2dvi',
    timeout = 60,
    useFancyQuotes = TRUE,
    verbose = FALSE,
    warn = 0,
    warning.length = 1000,
    nwarnings = 50,
    width = 80)

options <- function(...) {
    if(...() == 0L) {
        .Options
    }
    else {
        args <- list(...)
        if(is.null(names(args)) && length(args) == 1 && is.list(args[[1]]))
            args <- args[[1]]

        # split into named and unnamed parameters
        n <- names(args)
        if(is.null(n))
            n <- rep('', length(args))

        nargs <- args[n != '']
        .Options[names(nargs)] <<- strip(args)
        
        nuargs <- as.character(strip(args)[n == ''])
        uargs <- .Options[nuargs]
        names(uargs) <- nuargs
 
        c(nargs,uargs)
    }
}

