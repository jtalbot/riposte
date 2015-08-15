
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

        if(type == 0xf7) {
            # this seems to be 0 all the time
            int1 <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
            r <- refhook(.unserialize.character())
            refs[[length(refs)+1L]] <<- r
            return(r)
        }

        if(type == 0xf9) {
            # this is a namespace, but not sure how to read it.
            int1 <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
            int2 <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
            name <- .unserialize()
            version <- .unserialize()
            r <- internal::getRegisteredNamespace(name)
            refs[[length(refs)+1L]] <<- r
            return(r)
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
            idx <- as.integer(sexp[[3L]])+
                   256*as.integer(sexp[[2L]])+
                   65536*as.integer(sexp[[1L]])
            if(idx == 0L) {
                idx <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)
            }
            return(refs[[idx]])
        }

        if(as.integer(type) < 1 || as.integer(type) > 25) {
            print(type)
            .stop("Unsupported type in .unserialize (unknown)")
        }

        # if it has attributes and it's a pairlist
        # (or apparently some other things), they come first...
        if((flags & as.raw(0x02)) && 
            (type == 0x02 || type == 0x03 || type == 0x05 || type == 0x06)) {
            attrs <- .unserialize()
            #print('Attrs1')
            #print(attrs)
        }

        v <- switch(as.integer(type)+1,
            NULL,
            .unserialize.symbol(),
            .unserialize.pairlist(sexp),
            .unserialize.closure(),
            .unserialize.environment(),
            .unserialize.promise(sexp),
            .unserialize.language(),
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
            as.name('...'),
            .stop('Unsupported type in .unserialize (any)'),
            .unserialize.list(),
            .stop('Unsupported type in .unserialize (expressions vector)'),
            .unserialize.bytecode(),
            .unserialize.extptr(),
            .stop('Unsupported type in .unserialize (weak reference)'),
            .stop('Unsupported type in .unserialize (raw bytes)'),
            .unserialize.S4()
            )

        if(flags & as.raw(0x02)) {
            if(type != 0x02 && type != 0x03 && type != 0x05 && type != 0x06) {
                attrs <- .unserialize()
            }
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
        x <- as.call(list(as.name('function'), args, body, list()))
        x <- as.call(list(as.name('function'), args, body, .deparse(x)))
        promise('f', x, env, .getenv(NULL))
        f
    }

    .unserialize.environment <- function() {
        locked <- readBin(con, 'integer', 1L, 4L, TRUE, TRUE)

        enclosure <- .unserialize()
        e <- env.new(enclosure)
        refs[[length(refs)+1L]] <<- e

        frame <- .unserialize()
        tag <- .unserialize()
        e[names(tag)] <- strip(tag)
        e
    }

    .unserialize.promise <- function(sexp) {
        value <- .unserialize()
        expr <- .unserialize()
        value    
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

    .unserialize.pairlist <- function(sexp) {
        r <- list()
        names <- vector('character',0)

        while(sexp[[4L]] == 0x02) { 
            if(sexp[[3L]] & as.raw(0x04)) {
                names[[length(names)+1L]] <- .unserialize()
            }
            else {
                names[[length(names)+1L]] <- ""
            }

            r[[length(r)+1L]] <- .unserialize()
            sexp <- readBin(con, 'raw', 4L, 1L, TRUE, TRUE)
        }
        if(any(names != ""))
            attr(r, 'names') <- names
        attr(r, 'class') <- 'pairlist'
        r 
    }

    .unserialize.language <- function() {
        head <- .unserialize()
        tail <- .unserialize()
        r <- c.default(head, tail)
        attr(r, 'class') <- 'call'
        r
    }

    .unserialize.special <- function() {
        n <- as.name(.unserialize.string())
        r <- list(as.name('::'), as.name('primitive'), n)
        attr(r, 'class') <- 'call'
        r
    }

    .unserialize.builtin <- function() {
        n <- as.name(.unserialize.string())
        r <- list(as.name('::'), as.name('primitive'), n)
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

    .unserialize.extptr <- function() {
        protected <- .unserialize()
        tag <- .unserialize()
        r <- NULL
        refs[[length(refs)+1L]] <<- r
        r
    }

    .unserialize.S4 <- function() {
        NULL
    }

    .unserialize()
}

