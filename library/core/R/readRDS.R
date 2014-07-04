
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
            return(list())

        if(type == 0xff) {
            idx <- as.integer(flags)
            if(idx == 0L)
                idx <- readBin(con, 'integer', 1L, 4L, FALSE, TRUE)
            return(refs[[idx]])
        }

        if(as.integer(type) > 25) {
            .stop("Unsupported type in .unserialize (unknown)")
        }

        # if it has attributes and it's a pairlist
        # (or apparently some other things), they come first...
        if((flags & as.raw(0x02)) && 
            (type == 0x02 || type == 0x03 || type == 0x06)) {
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
            .stop('Unsupported type in .unserialize (external pointer)'),
            .stop('Unsupported type in .unserialize (weak reference)'),
            .stop('Unsupported type in .unserialize (raw bytes)'),
            .stop('Unsupported type in .unserialize (S4, non-vector)')
            )

        if(flags & as.raw(0x02)) {
            if(type != 0x02 && type != 0x03 && type != 0x06)
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
        r <- as.name(.unserialize())
        refs[[length(refs)+1L]] <<- r
        r
    }

    .unserialize.closure <- function() {
        env <- .unserialize()
        args <- .unserialize()
        body <- .unserialize()
        print(body)
        x <- as.call(list(as.name('function'), args, body, 'From compiled bytecode'))
        promise('f', x, env, .getenv(NULL))
        f
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
        readBin(con, 'logical', len, 4L, FALSE, TRUE)
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
        head <- .unserialize()
        tail <- .unserialize()
        r <- c.default(head, tail)
        attr(r, 'class') <- 'call'
        r
    }

    # for reasons beyond me, when parsing bytecodes, the format changes subtly.
    .unserialize.bc <- function() {
        sexp <- readBin(con, 'raw', 4L, 1L, TRUE, TRUE)
        type <- as.integer(sexp[[4L]])
        flags <- sexp[[3L]]
        
        # Byte code stuff that I don't care about...
        if(type == 0xf3) {
            int1 <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
            return(NULL)
        }

        if(type == 0xf4) {
            int1 <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
            sexp <- readBin(con, 'raw', 4L, 1L, TRUE, TRUE)
            type <- as.integer(sexp[[4L]])
            flags <- sexp[[3L]]
        }

        if(type == 0xf0) {
            type <- 0x06
            flags <- 0x02
        }

        if(type == 0xef) {
            type <- 0x02
            flags <- 0x02
        }

        if( type == 0x02 ) {
            if(flags & as.raw(0x02))
                attrs <- .unserialize()
           
            b <- list()
 
            a <- .unserialize()
            b[[1]] <- .unserialize.bc()
            q <- .unserialize.bc()
             
            if(!is.list(a)) {
                names(b) <- a
            }
            r <- c.default(b,q)
            attr(r, 'class') <- 'pairlist'
            return(r)
        }

        if( type == 0x06 ) {
            if(flags & as.raw(0x02))
                attrs <- .unserialize()
            a <- .unserialize()
            b <- .unserialize.bc()
            q <- .unserialize.bc()
           
            r <- c.default(b,q)
            attr(r, 'class') <- 'call'
            return(r)
        }

        if( type == 0x15 ) {
            return(.unserialize.bc.body())
        }
        
        .unserialize()
    }

    .unserialize.bc.body <- function() {
        bc <- .unserialize()
        # more stuff?
        # the body is in the first block here, but I have to parse
        # the rest to know where the end is
        r <- NULL
        len <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
        for( i in seq_len(len) ) {
            x <- .unserialize.bc()
            if(i == 1)
                r <- x
        }
        r
    }

    .unserialize.bytecode <- function() {
        # don't know why I need this length
        len <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
        .unserialize.bc.body()
    }

    .unserialize()
}

