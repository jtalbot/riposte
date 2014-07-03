
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

        if(type == 0xf1)
            return(baseenv())

        if(type == 0xf2)
            return(emptyenv())

        if(type == 0xf9) {
            # this is a namespace, but not sure how to read it.
            int1 <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
            int2 <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
            name <- .unserialize()
            version <- .unserialize()
            return(internal::getRegisteredNamespace(name))
        }

        if(type == 0xfa)
            return(internal::getRegisteredNamespace('base'))

        if(type == 0xfb)
            return(Nil)     # missing argument in Riposte

        if(type == 0xfd)
            return(globalenv())

        if(type == 0xfe)
            return(NULL)

        if(type == 0xff) {
            idx <- as.integer(flags)
            if(idx == 0L)
                idx <- readBin(con, 'integer', 1L, 4L, FALSE, TRUE)
            return(refs[[idx]])
        }

        if(as.integer(type) > 25) {
            .stop("Unsupported type in .unserialize (unknown)")
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
            .unserialize.language(flags),
            .unserialize.special(),
            .unserialize.builtin(),
            .unserialize.string(),
            .unserialize.logical(),
            .stop('Unsupported type in .unserialize (factors deprecated)'),
            .stop('Unsupported type in .unserialize (ordered factors deprecated)'),
            .unserialize.integer(),
            .unserialize.double(),
            .unserialize.complex(),
            .unserialize.character(),
            .stop('Unsupported type in .unserialize (...)'),
            .stop('Unsupported type in .unserialize (any)'),
            .unserialize.list(),
            .stop('Unsupported type in .unserialize (expressions vector)'),
            .unserialize.bytecode(),
            .stop('Unsupported type in .unserialize (extermal pointer)'),
            .stop('Unsupported type in .unserialize (weak reference)'),
            .stop('Unsupported type in .unserialize (raw bytes)'),
            .stop('Unsupported type in .unserialize (S4, non-vector)')
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

    .unserialize.closure <- function() {
        .stop("Unserializing closures is not yet implemented")
        a1 <- .unserialize()
        a2 <- .unserialize()
        a3 <- .unserialize()
        a4 <- .unserialize()
    }

    .unserialize.environment <- function() {
        locked <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
        enclosure <- .unserialize()
        frame <- .unserialize()
        tag <- .unserialize()
        e <- env.new(enclosure)
        e[names(tag)] <- strip(tag)
        e
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

    .unserialize.logical <- function() {
        len <- .length()
        readBin(con, 'logical', len, 4L, FALSE, FALSE)
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

    .unserialize.language <- function(flags) {
        r <- .unserialize.pairlist(as.raw(0x04))
        attr(r, 'class') <- 'call'
        r
    }

    .unserialize.bytecode <- function() {
        .stop("Unserializing bytecode is not yet supported")
        # don't know why I need this length
        len <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
        bc <- .unserialize()
        # more stuff?
        len <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
        for( i in seq_len(len) ) {
            a <- .unserialize()
        }
        NULL
    }

    .unserialize()
}

