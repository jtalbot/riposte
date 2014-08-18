
readChar <- function(con, nchars, useBytes) {
    if(useBytes) {
        if(!is.raw(con))
            con <- .readBin(con, nchars)
        else
            con <- con[seq_len(n*size)]

        .Riposte('rawToChar', con)
    }
    else {
        .stop('NYI: readChar with useBytes=FALSE')
    }
}
