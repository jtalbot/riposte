
.sunk_connections <- (function(...) list(...))() 

sink <- function(file, closeOnExit, message, split) {
    if(file == -1L) {
        if(length(.sunk_connections) > 0L)
            .sunk_connections <<- .sunk_connections[-length(.sunk_connections)]
    }
    else {
        .sunk_connections[[length(.sunk_connections)+1L]] <<- list(file, split)
    }
}

sink.number <- function(message) {
    length(.sunk_connections)
}

cat <- function(x, file, sep, fill, labels, append) {
    x <- paste(unlist(x, FALSE, FALSE), sep=sep, collapse=sep)
    if(.isTRUE(fill)) {
        x <- .pconcat(x, '\n')
    }
    if(identical(file, '')) {
        i <- length(.sunk_connections)
        while(i > 0L) {
            r <- .connection.cat(.sunk_connections[[i]][[1L]], x)
            if(!.sunk_connections[[i]][[2L]])
                return(r)
            i <- i-1L
        }
        .connection.cat(stdout(), x)
    }
    else {
        .connection.cat(file, x)
    }
}

.connection.cat <- function(con, x) UseMethod('.connection.cat', con)

