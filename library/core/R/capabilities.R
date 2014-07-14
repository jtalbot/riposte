
capabilities <- function(what=NULL) {
    r <- c(
        jpeg=FALSE,
        png=FALSE,
        tiff=FALSE,
        tcltk=FALSE,
        X11=FALSE,
        aqua=FALSE,
        `http/ftp`=FALSE,
        sockets=FALSE,
        libxml=FALSE,
        fifo=FALSE,
        cledit=FALSE,
        iconv=FALSE,
        NLS=FALSE,
        profmem=FALSE,
        cairo=FALSE)

    if(is.null(what))
        r
    else
        r[what]
}

capabilitiesX11 <- function() FALSE

