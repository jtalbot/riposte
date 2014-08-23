
tempfile <- function(pattern, tmpdir, fileext) {
    # TODO: actually generate a random hex number here
    paste(list(tmpdir,pattern,fileext), '', NULL)
}

tempdir <- function() {
    d <- Sys.getenv('TMPDIR', '')
    if(d == '')
        d <- Sys.getenv('TMP', '')
    if(d == '')
        d <- Sys.getenv('TEMP', '')
    if(d == '')
        d <- '/tmp'
    return(d)
}

