
unserializeFromConn <- function(con, refhook) {
    # read header
    header <- readChar(con, 2L, TRUE)
    if(header == 'X\n') {
        readBin(con, 'integer', 3L, 4L, TRUE, TRUE)
        .unserialize(con, refhook)
    }
    else {
        .stop('Non-XDR format not yet supported')
    }
}

.unserialize <- function(con, refhook) {

    refs <- list()

    .unserialize <- function() {
        .unserialize.dispatch(readBin(con, 'raw', 4L, 1L, TRUE, TRUE))
    }

    .unserialize.dispatch <- function(sexp) {
        type <- as.integer(sexp[[4L]])
        flags <- sexp[[3L]]

        if(type == 0xfe)
            return(NULL)

        if(type == 0xff) {
            idx <- as.integer(flags)
            if(idx == 0L)
                idx <- readBin(con, 'integer', 1L, 4L, FALSE, TRUE)
            return(refs[[idx]])
        }

        # if it has attributes and it's a pairlist, they come first...
        if((flags & as.raw(0x02)) && type == 0x02) {
            attrs <- .unserialize()
        }

        v <- switch(as.integer(type)+1,
            Nil,
            .unserialize.symbol(),
            .unserialize.pairlist(flags),
            .unserialize.closure(),
            .unserialize.environment(),
            .unserialize.promise(),
            .unserialize.language(),
            .unserialize.special(),
            .unserialize.builtin(),
            .unserialize.string(),
            .unserialize.logical(),
            .stop('Unsupported type in .unserialize'),
            .stop('Unsupported type in .unserialize'),
            .unserialize.integer(),
            .unserialize.double(),
            .unserialize.complex(),
            .unserialize.character(),
            .stop('Unsupported type in .unserialize'),
            .stop('Unsupported type in .unserialize'),
            .unserialize.list(),
            .stop('Unsupported type in .unserialize')
            )

        if(flags & as.raw(0x02)) {
            if(type != 0x02)
                attrs <- .unserialize()
            attributes(v) <- attrs
        }

        v    
    }

    .length <- function() {
        len <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
        if(len < 0L) {
            upper <- readBin(con, 'integer', 1L, 4L, FALSE, TRUE)
            lower <- readBin(con, 'integer', 1L, 4L, FALSE, TRUE)
        len <- upper * 4294967296 + lower
    }
    len
    }

    .unserialize.symbol <- function() {
        r <- as.vector(.unserialize(), 'symbol')
        refs[[length(refs)+1L]] <<- r
        r
    }

    .unserialize.string <- function() {
        len <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
        if(len < 0L)
            NA_character_
        else
            readChar(con, len, TRUE)
    }

    .unserialize.character <- function() {
        len <- .length()
        r <- vector('character', len)
        for( i in seq_len(len) ) {
            r[[i]] <- as.character(.unserialize())
        }
        r
    }

    .unserialize.integer <- function() {
        len <- .length()
        readBin(con, 'integer', len, 4L, TRUE, TRUE)
    }

    .unserialize.double <- function() {
        len <- .length()
        readBin(con, 'double', len, 8L, TRUE, TRUE)
    }

    .unserialize.list <- function() {
        len <- .length()
        r <- list()
        for( i in seq_len(len) ) {
            r[[i]] <- .unserialize()
        }
        r
    }

    .unserialize.pairlist <- function(flags) {
        r <- list()
        names <- vector('character',0)
       
        if(flags & as.raw(0x04)) {
            names[[length(names)+1L]] <- .unserialize()
        }
        else {
            names[[length(names)+1L]] <- ""
        }
     
        sexp <- readBin(con, 'raw', 4L, 1L, TRUE, TRUE)
        while(sexp[[4L]] != 0xfe) {
            # unserialize CAR
            r[[length(r)+1L]] <- .unserialize.dispatch(sexp)
            sexp <- readBin(con, 'raw', 4L, 1L, TRUE, TRUE)
            # if CDR is also a pairlist, flatten into this list
            if(sexp[[4L]] == 0x02) {
                flags <- sexp[[3L]]

                if(flags & as.raw(0x04)) {
                    names[[length(names)+1L]] <- .unserialize()
                }
                else {
                    names[[length(names)+1L]] <- ""
                }

                sexp <- readBin(con, 'raw', 4L, 1L, TRUE, TRUE)
            }
        }
        if(any(names != ""))
            attr(r, 'names') <- names
        attr(r, 'class') <- 'pairlist'
        r 
    }

    .unserialize()
}
